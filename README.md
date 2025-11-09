# Solar Forecast ML - Version 8 "Sarpeidon" 🌟

**First machine learning solar forecasting for Home Assistant**

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![GitHub release](https://img.shields.io/github/release/Zara-Toorox/ha-solar-forecast-ml.svg)](https://github.com/Zara-Toorox/ha-solar-forecast-ml/releases)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

<a href='https://ko-fi.com/Q5Q41NMZZY' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://ko-fi.com/img/githubbutton_sm.svg' border='0' alt='Buy Me a Coffee ' /></a>

Imagine your smart home not just reacting, but **predicting** - using machine learning that adapts daily to your unique solar installation. That's exactly what **Solar Forecast ML** does, and with Version 8 "Sarpeidon" we reach a new dimension of precision.

## What Makes This Integration Special?

Unlike simple weather API services, Solar Forecast ML **learns from your real data**. This means: The longer the integration runs, the more accurate the predictions become - adapted to your specific roof orientation, local weather conditions, shading, and even seasonal characteristics.

### Core: Self-Learning ML Model

The integration collects daily weather data and compares it with your actual solar yield. Through advanced machine learning algorithms, it recognizes patterns and continuously optimizes its predictions. The result? **Precise forecasts that are better than any weather service alone.**


## Solar ML Features

### 📊 Core Functions
- **3-day yield forecasts** for today, tomorrow and the day after tomorrow with automatic ML optimization
- **Hourly predictions** for detailed planning (optional)
- **Intelligent weather integration** with automatic fallback on failures
- **Self-learning system** - gets more precise every day without your intervention
- **Real-time accuracy tracking** - see the precision of your forecasts
- **No RAM usage** - everything stored on your disk
- **100% private** - everything stays on your disk, everything is calculated on your system

### 🎯 Production Analytics
- **Peak hour detection** - when do you produce the most?
- **Production time calculation** - optimized from sunrise to sunset
- **Average yield** over any time period
- **Self-sufficiency calculation** - how independent are you really?
- **Deviation analysis** - yesterday vs. forecast in direct comparison

### 🔬 Diagnostics & Transparency (20+ Sensors)
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
- **Update Forecast** - manual update on demand
- **Retrain Model** - complete re-training with all historical data
- **Calibrate Model** - fine-tuning for quick adjustments
- **Clear Errors** - reset on problems

### 🔔 Smart Notifications
Stay informed with configurable notifications for:
- Integration startup
- New forecast updates
- Ongoing ML training processes
- Successfully completed training

---

## 🔋 Optional Extension: Battery Management (v8.2.0)

**Completely separate module** - No interference with Solar ML!

### Battery Features
- **Battery monitoring** - SOC, power, temperature
- **Charge/discharge tracking** - Daily statistics
- **Solar forecast integration** - Expected battery charge from solar
- **Runtime calculation** - Remaining autonomy
- **Efficiency tracking** - Round-trip efficiency (default: 90%)
- **Self-sufficiency with battery** - Extended self-sufficiency calculation
- **Grid monitoring** - Import/export tracking
- **Direct consumption** - Solar direct use vs. battery

### Battery Sensors (14 Sensors)
**Basic Monitoring:**
- `sensor.battery_soc` - State of Charge (%)
- `sensor.battery_power` - Current power (W)
- `sensor.battery_charge_today` - Charge today (kWh)
- `sensor.battery_discharge_today` - Discharge today (kWh)
- `sensor.battery_expected_charge_solar` - Expected charge from solar (kWh)

**Analytics:**
- `sensor.battery_charge_from_solar` - Charge from solar (kWh)
- `sensor.battery_charge_from_grid` - Charge from grid (kWh)
- `sensor.battery_runtime_remaining` - Remaining runtime (h)
- `sensor.battery_efficiency` - Efficiency (%)
- `sensor.autarkie_with_battery` - Self-sufficiency with battery (%)
- `sensor.self_consumption_with_battery` - Self-consumption with battery (%)
- `sensor.grid_export_today` - Grid export today (kWh)
- `sensor.grid_import_today` - Grid import today (kWh)
- `sensor.direct_solar_consumption` - Direct solar consumption (kWh)

### Battery Configuration
**Settings** → **Devices & Services** → **Solar Forecast ML** → **Configure**

- **Enable Battery Management** - Enable battery features
- **Battery Capacity** - Battery capacity (0.5-1000 kWh, default: 10.0)
- **Battery SOC Entity** - State of Charge sensor (%)
- **Battery Power Entity** - Current charge/discharge power (W)
- **Battery Charge Today** - Daily charge (kWh)
- **Battery Discharge Today** - Daily discharge (kWh)
- **Battery Temperature** - Optional: temperature sensor

### Battery Automation Example
```yaml
automation:
  - alias: "Intelligent battery strategy"
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

---

## 💰 Optional Extension: Electricity Prices (v8.2.0)

**Free aWATTar API** - No registration required!

### Electricity Price Features
- **Real-time prices** - Current and next hour (ct/kWh)
- **Price statistics** - Min/max/average (today/week)
- **Cheapest hours** - Top 3 cheapest time slots
- **Charging recommendation** - "Charge now" or "Wait until XX:00"
- **Savings tracking** - Daily savings in EUR
- **aWATTar integration** - Free, DE/AT support
- **EPEX Spot data** - Updated daily at 13:00

### Electricity Sensors (10 Sensors)
- `sensor.electricity_price_current` - Current price (ct/kWh)
- `sensor.electricity_price_next_hour` - Price next hour (ct/kWh)
- `sensor.electricity_price_avg_today` - Average today (ct/kWh)
- `sensor.electricity_price_avg_week` - Average week (ct/kWh)
- `sensor.electricity_price_min_today` - Minimum today (ct/kWh)
- `sensor.electricity_price_max_today` - Maximum today (ct/kWh)
- `sensor.electricity_cheapest_hour_today` - Cheapest hour
- `sensor.electricity_most_expensive_hour_today` - Most expensive hour
- `sensor.electricity_charging_recommendation` - Charging recommendation
- `sensor.electricity_savings_today` - Savings today (€)

### aWATTar API Details
- **No registration** - Public API
- **100 requests/day** - Fair use policy
- **Data source** - EPEX Spot exchange prices
- **Update** - Daily at 13:00 (day-ahead prices)
- **Countries** - Germany (DE) and Austria (AT)
- **Format** - EUR/MWh → automatically converted to ct/kWh

### Electricity Configuration
**Settings** → **Devices & Services** → **Solar Forecast ML** → **Configure**

- **Enable Electricity Prices** - Enable electricity price features
- **Country** - Country (DE or AT, default: DE)
- **No API key required** - aWATTar API is free

### Electricity Automation Example
```yaml
automation:
  - alias: "Charge battery at cheap prices"
    trigger:
      - platform: state
        entity_id: sensor.electricity_charging_recommendation
        to: "Charge now"
    condition:
      - condition: numeric_state
        entity_id: sensor.battery_soc
        below: 80
    action:
      - service: switch.turn_on
        target:
          entity_id: switch.battery_grid_charging
      - service: notify.mobile_app
        data:
          message: "Cheap electricity price! Battery charging ({{ states('sensor.electricity_price_current') }} ct/kWh)"
```

### Combined Automation (Solar + Battery + Electricity)
```yaml
automation:
  - alias: "Intelligent daily energy report"
    trigger:
      - platform: time
        at: "20:00:00"
    action:
      - service: notify.telegram
        data:
          message: >
            📊 Daily energy balance:
            
            ☀️ Solar: {{ states('sensor.solar_yield_today') }} kWh
            🔋 Battery charge: {{ states('sensor.battery_charge_today') }} kWh
            🔌 Grid import: {{ states('sensor.grid_import_today') }} kWh
            
            💰 Savings: {{ states('sensor.electricity_savings_today') }} €
            📈 Self-sufficiency: {{ states('sensor.autarkie_with_battery') }}%
            
            ⚡ Forecast deviation: {{ states('sensor.yesterday_deviation') }}%
```

### Troubleshooting (Extensions)
**Battery sensors missing:**
- Enable Battery Management in configuration
- Configure all battery entities
- Restart Home Assistant after configuration changes

**Electricity prices not available:**
- Enable Electricity Prices in configuration
- Select country (DE/AT)
- Check internet connection (aWATTar API access)
- Check after 13:00 (daily update)

---

## Installation

### HACS (Recommended)
1. Open HACS → Integrations
2. Three dots top right → Custom repositories
3. Add URL: `https://github.com/Zara-Toorox/ha-solar-forecast-ml`
4. Category: Integration
5. Download "Solar Forecast ML"
6. Restart Home Assistant

### Manual Installation
1. Download latest release from GitHub
2. Extract `solar_forecast_ml` folder
3. Copy to `custom_components`
4. Restart Home Assistant

## Configuration

### Quick Setup in 3 Steps

1. **Settings** → **Devices & Services**
2. **Add Integration**
3. Search **Solar Forecast ML**

### Required Sensors
- **Weather entity** (e.g. DWD)
- **Power sensor** in watts
- **Daily yield** in kWh (must reset daily at midnight!)

### Optional Solar Sensors
- **Plant capacity** (kWp) for more precise calculations
- **External sensors** (temperature, wind, rain, UV, lux, humidity)
- **Consumption sensor** for self-sufficiency calculation

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

### Example 2: Dynamic Energy Management
```yaml
automation:
  - alias: "Energy strategy based on forecast"
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
              - service: script.energy_saving_mode
          - conditions:
              - condition: numeric_state
                entity_id: sensor.solar_forecast_today
                above: 20
            sequence:
              - service: script.high_consumption_mode
```

### Example 3: Energy-Intensive Devices at Peak Time
```yaml
automation:
  - alias: "High consumption at peak production"
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
  - alias: "Daily energy report"
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
            Deviation: {{ states('sensor.yesterday_deviation') }}%
            Production time: {{ states('sensor.production_time') }}
```

## Available Sensors

### Main Forecast Sensors
- `sensor.solar_forecast_heute` - Today's forecast
- `sensor.solar_forecast_morgen` - Tomorrow's forecast
- `sensor.solar_forecast_3days` - Day after tomorrow
- `sensor.peak_production_hour` - Hour with highest production
- `sensor.production_time` - Active production time
- `sensor.average_yield` - Average yield
- `sensor.autarkie` - Self-sufficiency percentage

### Accuracy & Diagnostics
- `sensor.solar_accuracy` - Model accuracy
- `sensor.yesterday_deviation` - Yesterday's deviation
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
- Feature vectors, weights, model metrics, training samples and more

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

**Settings** → **Devices & Services** → **Solar Forecast ML** → **Configure**

### Solar ML Options
- **Update interval** - How often forecasts update (300-86400 seconds)
- **Diagnostic mode** - Enable extended diagnostic sensors
- **Hourly forecasts** - Enable hourly predictions
- **Notifications** - Configure startup, forecast, learning notifications

## Troubleshooting

### Integration not starting
- Check that all required sensors exist
- Daily yield sensor must reset at midnight
- Check Home Assistant logs for errors

### Inaccurate predictions
- Wait at least 10 days for ML training
- Verify weather entity provides reliable data
- Check that power and yield sensors report correct values
- Use "Retrain Model" button to rebuild ML model

### Missing sensors
- Enable diagnostic mode in configuration
- Restart Home Assistant after configuration changes
- Check entity registry for disabled sensors

## Technical Highlights

- **Modular architecture** - Cleanly separated manager classes
- **Async/await everywhere** - No blocking operations
- **Comprehensive error handling** - Integration runs stable, even with sensor failures
- **Data persistence** - All ML models and historical data remain after restarts
- **Performance optimized** - Intelligent caching and minimal update intervals
- **Best practice compliant** - Follows all Home Assistant development guidelines

### Version 8 Core Features:
- Refined ML algorithms with better convergence
- Optimized feature weighting
- More robust error handling
- Extended diagnostic sensors
- Improved weather service integration with intelligent retry system

### Version 8.2.0 Optional Extensions:
- **Complete Battery Management** - Monitoring, analytics, grid tracking
- **Electricity price integration** with aWATTar API (free, DE/AT)
- **Charging optimization** based on exchange prices
- **Separate modules** - No interference with Solar ML Core

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
