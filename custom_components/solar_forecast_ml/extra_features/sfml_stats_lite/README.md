# SFML Stats Lite Dashboard

[![Version](https://img.shields.io/badge/version-2.2.0-blue.svg)](https://github.com/Zara-Toorox/sfml_stats_lite)
[![HACS](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://hacs.xyz/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-green.svg)](LICENSE)

<a href='https://ko-fi.com/Q5Q41NMZZY' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://ko-fi.com/img/githubbutton_sm.svg' border='0' alt='Buy Me a Coffee ' /></a>

**Energy Monitoring Dashboard for Home Assistant**

Dashboard zur Visualisierung von Solarproduktion, Batteriespeicher, Netzverbrauch und Energiekosten. Funktioniert mit [Solar Forecast ML](https://github.com/Zara-Toorox/ha-solar-forecast-ml) und [Grid Price Monitor](https://github.com/Zara-Toorox/ha-solar-forecast-ml/tree/main/custom_components/solar_forecast_ml/extra_features/grid_price_monitor).

---

## What Makes This Dashboard Special?

SFML Stats Lite Dashboard provides a unified view of your entire energy ecosystem:

- **Real-time Visualization:** See exactly where your energy is flowing - from solar panels to battery, house, and grid
- **Forecast Integration:** Compare actual production against Solar Forecast ML predictions
- **Cost Tracking:** Monitor energy costs with fixed or dynamic pricing support
- **Automated Reports:** Generate beautiful weekly and monthly charts automatically
- **Multi-String Support:** Track up to 4 separate panel groups individually
- **Weather Correlation:** Overlay weather data on your energy charts

---

## Features

- Real-time energy flow visualization
- Solar production analytics with forecast comparison
- Battery state of charge and charge/discharge tracking
- Grid import/export monitoring
- Cost calculation with fixed or dynamic pricing
- Automated weekly and monthly report generation
- Support for up to 4 separate solar panel strings
- Weather integration for chart annotations
- Dark and light theme support

---

## Screenshots

*Coming soon*

---

## Installation

### HACS (Recommended)

1. Open HACS in Home Assistant
2. Click on "Integrations"
3. Click the three dots menu and select "Custom repositories"
4. Add this repository URL and select "Integration" as category
5. Search for "SFML Stats Lite" and install
6. Restart Home Assistant
7. Go to Settings > Devices & Services > Add Integration > SFML Stats Lite

### Manual Installation

1. Download the latest release
2. Copy the `sfml_stats_lite` folder to `custom_components/` in your Home Assistant config directory
3. Restart Home Assistant
4. Go to Settings > Devices & Services > Add Integration > SFML Stats Lite

---

## Sensor Configuration Guide

### General Notes

All sensors are optional. The integration works with incomplete configuration but will only display available data.

Power sensors can be specified in Watts (W) or Kilowatts (kW). The integration detects the unit automatically.

Energy sensors should be in Kilowatt-hours (kWh).

---

## Step 1: Basic Settings

No sensors required. This step contains configuration options for automatic chart generation and color theme selection.

---

## Step 2: Energy Flow Sensors

### sensor_solar_power
Current total power output of the photovoltaic system.
- Unit: W
- Source: Inverter integration
- Example: `sensor.inverter_pv_power`

### sensor_solar_to_house
Portion of solar power consumed directly in the house.
- Unit: W
- Source: Inverter or energy management system
- Note: Not all inverters provide this value. If unavailable, it can often be calculated from other values.

### sensor_solar_to_battery
Solar power flowing into the battery.
- Unit: W
- Source: Battery management system or hybrid inverter
- Note: Only relevant for systems with battery storage.

### sensor_grid_to_house
Power currently drawn from the grid.
- Unit: W
- Source: Smart meter or inverter
- Note: Some systems only show total grid consumption without breakdown.

### sensor_grid_to_battery
Grid power used to charge the battery.
- Unit: W
- Source: Battery management system
- Note: Relevant for systems that allow grid charging.

### sensor_house_to_grid
Power fed into the grid.
- Unit: W
- Source: Smart meter or inverter
- Note: Shows current surplus being exported.

### sensor_smartmeter_import
Total grid consumption from smart meter.
- Unit: kWh
- Source: Smart meter integration
- Note: Can be a meter reading (continuously increasing) or a daily counter.

### sensor_smartmeter_export
Total export according to smart meter.
- Unit: kWh
- Source: Smart meter integration
- Note: Corresponds to the feed-in meter.

---

## Step 3: Battery Sensors

### sensor_battery_soc
Current battery state of charge in percent.
- Unit: Percent (0-100)
- Source: Battery management system
- Example: `sensor.battery_state_of_charge`

### sensor_battery_power
Current charging or discharging power of the battery.
- Unit: W
- Source: Battery management system
- Note: Positive values typically mean charging, negative values discharging. Some systems use reversed logic.

### sensor_battery_to_house
Discharge power flowing to the house.
- Unit: W
- Source: Battery management system or hybrid inverter

### sensor_battery_to_grid
Discharge power being fed into the grid.
- Unit: W
- Source: Battery management system
- Note: Only relevant if the system can export battery power.

### sensor_home_consumption
Total current power consumption of the household.
- Unit: W
- Source: Energy management system or custom calculation
- Note: Sum of all sources (solar, battery, grid).

---

## Step 4: Statistics Sensors

### sensor_solar_yield_daily
Daily yield of the photovoltaic system.
- Unit: kWh
- Source: Inverter or Home Assistant Utility Meter
- Note: Should reset to zero at midnight.

### sensor_grid_import_daily
Daily grid consumption.
- Unit: kWh
- Source: Smart meter or Home Assistant Utility Meter
- Note: Should reset to zero at midnight.

### sensor_grid_import_yearly
Yearly grid consumption.
- Unit: kWh
- Source: Smart meter or Home Assistant Utility Meter
- Note: Used for annual statistics.

### sensor_battery_charge_solar_daily
Daily battery charge from solar power.
- Unit: kWh
- Source: Battery management system or custom integration

### sensor_battery_charge_grid_daily
Daily battery charge from grid power.
- Unit: kWh
- Source: Battery management system

### sensor_price_total
Total cost for grid consumption.
- Unit: Currency (EUR, USD, etc.)
- Source: Custom calculation or energy cost integration

### weather_entity
Weather entity for weather display in charts.
- Type: Weather entity (not sensor)
- Source: Any Home Assistant weather integration
- Example: `weather.home`

---

## Step 5: Panel Sensors

The integration supports up to four separate strings or panel groups. Each group has three fields.

### panel_name (1-4)
Custom name for the string.
- Type: Text
- Examples: South, East, West, Garage, Carport
- Default: String 1, String 2, String 3, String 4

### sensor_panel_power (1-4)
Current power output of this string.
- Unit: W
- Source: Inverter with MPPT tracking
- Note: Not all inverters provide individual string values.

### sensor_panel_max_today (1-4)
Peak power reached by this string on the current day.
- Unit: W
- Source: Custom calculation or inverter
- Note: Can be created with a Home Assistant Statistics helper using maximum function.

---

## Step 6: Billing

No sensors required. This step configures parameters for cost calculation.

The billing period defines from which day and month the annual energy balance is calculated. This typically corresponds to the start of your electricity contract.

The price mode can be dynamic (via a separate integration like Grid Price Monitor) or set as a fixed value in cents per kilowatt-hour.

---

## Practical Tips

### Creating Missing Sensors

If a required sensor is not directly available, it can often be created in Home Assistant:

**For daily counters:** Use the Utility Meter helper with daily reset.

**For maximum values:** Use the Statistics helper with the Maximum function.

**For calculated values:** Create a Template sensor.

### Typical Sensor Sources by Manufacturer

| Manufacturer | Integration |
|--------------|-------------|
| Fronius | Native integration provides most values directly |
| SMA | Modbus or SMA Energy Meter integration |
| Huawei | FusionSolar integration or Modbus |
| Kostal | Piko or Plenticore integration |
| Growatt | Growatt Server integration |
| Sungrow | Modbus integration |
| Enphase | Envoy integration provides panel data |

### Smart Meter Connection Options

- **ISKRA or similar:** IR reading head with ESPHome
- **P1 Port:** For Dutch and Belgian meters
- **Shelly EM/3EM:** As intermediate solution
- **Direct Modbus:** For modern meters

### Setting Up Utility Meters

For daily counters in `configuration.yaml`:

```yaml
utility_meter:
  solar_daily:
    source: sensor.inverter_total_energy
    cycle: daily
  grid_import_daily:
    source: sensor.smartmeter_import_total
    cycle: daily
```

### Template Sensor for Home Consumption

If no direct sensor is available:

```yaml
template:
  - sensor:
      - name: "Home Consumption"
        unit_of_measurement: "W"
        state: >
          {{ states('sensor.solar_power')|float(0)
             + states('sensor.grid_import')|float(0)
             - states('sensor.grid_export')|float(0) }}
```

---

## Accessing the Dashboard

After installation and configuration, access the dashboard at:

```
http://YOUR_HOME_ASSISTANT:8123/api/sfml_stats_lite/dashboard
```

Or add it to your sidebar via the Home Assistant configuration.

---

## Requirements

- Home Assistant 2024.1.0 or newer
- Python packages: aiofiles (installed automatically)

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox

---

## Support

- [GitHub Issues](https://github.com/Zara-Toorox/sfml_stats_lite/issues)
- [Documentation](https://github.com/Zara-Toorox/sfml_stats_lite)
