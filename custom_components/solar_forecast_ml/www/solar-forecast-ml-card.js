/**
 * Solar Forecast ML - Energy Flow Card v9.0.0
 *
 * Custom Lovelace Card for Home Assistant
 * Visualizes energy flows from Solar Forecast ML Battery Management
 *
 * Copyright (C) 2025 Zara-Toorox
 * License: AGPL-3.0
 */

class SolarForecastMLCard extends HTMLElement {
  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
    this._config = {};
    this._entities = {};
  }

  /**
   * Set configuration from YAML
   */
  setConfig(config) {
    if (!config) {
      throw new Error('Invalid configuration');
    }
    this._config = {
      title: config.title || 'Solar Forecast ML',
      show_solar: config.show_solar !== false,
      show_battery: config.show_battery !== false,
      show_grid: config.show_grid !== false,
      show_energy_flows: config.show_energy_flows !== false,
      show_forecast: config.show_forecast !== false,
      show_price: config.show_price !== false,
      compact_mode: config.compact_mode || false,
      animation_speed: config.animation_speed || 1.0,
      language: config.language || 'de',
      ...config
    };
  }

  /**
   * Set Home Assistant object
   */
  set hass(hass) {
    this._hass = hass;
    this._detectEntities();
    this._render();
  }

  /**
   * Auto-detect Solar Forecast ML entities
   */
  _detectEntities() {
    if (!this._hass) return;

    const states = this._hass.states;
    const entities = Object.keys(states);

    // Find Battery Management sensors (v9.0.0)
    this._entities = {
      // Power sensors (W)
      solarProduction: this._findEntity(entities, ['battery_management.*solar.*production', 'solar_production', 'pv_power', 'pv.*total']),
      inverterOutput: this._findEntity(entities, ['battery_management.*inverter.*output', 'inverter_output', 'inverter_power']),
      batteryPower: this._findEntity(entities, ['battery_management.*batterie.*leistung', 'battery_management.*battery.*power', 'battery_power']),
      batterySoc: this._findEntity(entities, ['battery_management.*batterie.*ladezustand', 'battery_management.*battery.*soc', 'battery.*soc', 'state_of_charge']),
      gridImport: this._findEntity(entities, ['battery_management.*grid.*import', 'battery_management.*netz.*import', 'grid_import', 'netzbezug']),
      gridExport: this._findEntity(entities, ['battery_management.*grid.*export', 'battery_management.*netz.*export', 'grid_export']),
      houseConsumption: this._findEntity(entities, ['battery_management.*house.*consumption', 'battery_management.*haus.*verbrauch', 'house_consumption', 'load_power']),

      // Energy flow sensors (kWh) - v9.0.0
      solarToHouse: this._findEntity(entities, ['battery_management.*solar.*to.*house', 'battery_management.*solar.*haus', 'solar_to_house']),
      solarToBattery: this._findEntity(entities, ['battery_management.*solar.*to.*battery', 'battery_management.*solar.*akku', 'solar_to_battery']),
      solarToGrid: this._findEntity(entities, ['battery_management.*solar.*to.*grid', 'battery_management.*solar.*netz', 'solar_to_grid']),
      batteryToHouse: this._findEntity(entities, ['battery_management.*battery.*to.*house', 'battery_management.*akku.*haus', 'battery_to_house']),
      gridToHouse: this._findEntity(entities, ['battery_management.*grid.*to.*house', 'battery_management.*netz.*haus', 'grid_to_house']),
      gridToBattery: this._findEntity(entities, ['battery_management.*grid.*to.*battery', 'battery_management.*netz.*akku', 'grid_to_battery']),

      // Additional sensors
      batteryRuntime: this._findEntity(entities, ['battery_management.*batterie.*restlaufzeit', 'battery_management.*battery.*runtime', 'battery.*runtime', 'restlaufzeit']),
      batteryEfficiency: this._findEntity(entities, ['battery_management.*batterie.*effizienz', 'battery_management.*battery.*efficiency', 'battery.*efficiency', 'effizienz']),
      forecastTomorrow: this._findEntity(entities, ['forecast.*tomorrow', 'prognose.*morgen', 'solar.*forecast.*tomorrow']),
      electricityPrice: this._findEntity(entities, ['electricity.*price', 'strompreis', 'awattar.*price']),
    };

    console.log('Solar Forecast ML Card - Detected entities:', this._entities);

    // Log current values for debugging
    console.log('Solar Forecast ML Card - Current values:');
    for (const [key, entityId] of Object.entries(this._entities)) {
      if (entityId) {
        const value = this._getState(entityId);
        console.log(`  ${key} (${entityId}): ${value}`);
      } else {
        console.log(`  ${key}: NOT FOUND`);
      }
    }
  }

  /**
   * Find entity by patterns
   */
  _findEntity(entities, patterns) {
    for (const pattern of patterns) {
      const regex = new RegExp(pattern, 'i');
      const found = entities.find(e => regex.test(e));
      if (found) return found;
    }
    return null;
  }

  /**
   * Get entity state value
   */
  _getState(entityId) {
    if (!entityId || !this._hass) return null;
    const state = this._hass.states[entityId];
    if (!state) {
      console.warn(`Solar Forecast ML Card - Entity not found: ${entityId}`);
      return null;
    }

    // Check for unavailable/unknown states
    if (state.state === 'unavailable' || state.state === 'unknown' || state.state === 'none') {
      console.warn(`Solar Forecast ML Card - Entity ${entityId} is ${state.state}`);
      return null;
    }

    const value = parseFloat(state.state);
    if (isNaN(value)) {
      console.warn(`Solar Forecast ML Card - Invalid numeric value for ${entityId}: ${state.state}`);
      return null;
    }

    return value;
  }

  /**
   * Get entity state object
   */
  _getStateObj(entityId) {
    if (!entityId || !this._hass) return null;
    return this._hass.states[entityId];
  }

  /**
   * Format power value
   */
  _formatPower(watts) {
    if (watts === null || watts === undefined) return '—';
    const absWatts = Math.abs(watts);

    if (absWatts >= 1000) {
      return `${(watts / 1000).toFixed(2)} kW`;
    }
    return `${watts.toFixed(0)} W`;
  }

  /**
   * Format energy value
   */
  _formatEnergy(kwh) {
    if (kwh === null || kwh === undefined) return '—';
    return `${kwh.toFixed(2)} kWh`;
  }

  /**
   * Format percentage
   */
  _formatPercent(value) {
    if (value === null || value === undefined) return '—';
    return `${value.toFixed(0)}%`;
  }

  /**
   * Get translations
   */
  _t(key) {
    const translations = {
      de: {
        solar: 'Solar',
        battery: 'Akku',
        grid: 'Netz',
        house: 'Haus',
        inverter: 'Wechselrichter',
        charging: 'Laden',
        discharging: 'Entladen',
        idle: 'Bereit',
        soc: 'Ladung',
        runtime: 'Restzeit',
        efficiency: 'Effizienz',
        forecast_tomorrow: 'Prognose Morgen',
        price: 'Strompreis',
        energy_flows: 'Energieflüsse Heute',
        solar_to_house: 'Solar → Haus',
        solar_to_battery: 'Solar → Akku',
        solar_to_grid: 'Solar → Netz',
        battery_to_house: 'Akku → Haus',
        grid_to_house: 'Netz → Haus',
        grid_to_battery: 'Netz → Akku',
      },
      en: {
        solar: 'Solar',
        battery: 'Battery',
        grid: 'Grid',
        house: 'House',
        inverter: 'Inverter',
        charging: 'Charging',
        discharging: 'Discharging',
        idle: 'Idle',
        soc: 'Charge',
        runtime: 'Runtime',
        efficiency: 'Efficiency',
        forecast_tomorrow: 'Forecast Tomorrow',
        price: 'Electricity Price',
        energy_flows: 'Energy Flows Today',
        solar_to_house: 'Solar → House',
        solar_to_battery: 'Solar → Battery',
        solar_to_grid: 'Solar → Grid',
        battery_to_house: 'Battery → House',
        grid_to_house: 'Grid → House',
        grid_to_battery: 'Grid → Battery',
      }
    };

    const lang = this._config.language || 'de';
    return translations[lang][key] || key;
  }

  /**
   * Render the card
   */
  _render() {
    if (!this._hass) return;

    // Get current values
    const solar = this._getState(this._entities.solarProduction) || 0;
    const inverter = this._getState(this._entities.inverterOutput) || 0;
    const batteryPower = this._getState(this._entities.batteryPower) || 0;
    const batterySoc = this._getState(this._entities.batterySoc) || 0;
    const gridImport = this._getState(this._entities.gridImport) || 0;
    const gridExport = this._getState(this._entities.gridExport) || 0;
    const house = this._getState(this._entities.houseConsumption) || 0;

    // Energy flows
    const solarToHouse = this._getState(this._entities.solarToHouse) || 0;
    const solarToBattery = this._getState(this._entities.solarToBattery) || 0;
    const solarToGrid = this._getState(this._entities.solarToGrid) || 0;
    const batteryToHouse = this._getState(this._entities.batteryToHouse) || 0;
    const gridToHouse = this._getState(this._entities.gridToHouse) || 0;
    const gridToBattery = this._getState(this._entities.gridToBattery) || 0;

    // Battery status
    const batteryStatus = batteryPower > 50 ? 'charging' :
                         batteryPower < -50 ? 'discharging' : 'idle';

    const batteryColor = batteryStatus === 'charging' ? '#4CAF50' :
                         batteryStatus === 'discharging' ? '#FF9800' : '#757575';

    this.shadowRoot.innerHTML = `
      <style>
        ${this._getStyles()}
      </style>

      <ha-card>
        <div class="card-header">
          <div class="name">${this._config.title}</div>
          <div class="version">v9.0.0</div>
        </div>

        <div class="card-content">
          <!-- Power Flow Diagram -->
          <div class="power-flow">
            <!-- Solar -->
            <div class="flow-node solar">
              <ha-icon icon="mdi:solar-power"></ha-icon>
              <div class="node-label">${this._t('solar')}</div>
              <div class="node-value">${this._formatPower(solar)}</div>
              <div class="node-sublabel">DC</div>
            </div>

            <!-- Flow Arrow Down -->
            <div class="flow-arrow ${solar > 50 ? 'active' : ''}">
              <div class="arrow-line"></div>
              <div class="flow-particles"></div>
            </div>

            <!-- Inverter -->
            <div class="flow-node inverter">
              <ha-icon icon="mdi:flash"></ha-icon>
              <div class="node-label">${this._t('inverter')}</div>
              <div class="node-value">${this._formatPower(inverter)}</div>
              <div class="node-sublabel">AC</div>
            </div>

            <!-- Flow Arrows to destinations -->
            <div class="flow-split">
              <div class="flow-arrow-split ${inverter > 50 ? 'active' : ''}"></div>
            </div>

            <!-- Destination Nodes -->
            <div class="flow-destinations">
              <!-- House -->
              <div class="flow-node house">
                <ha-icon icon="mdi:home"></ha-icon>
                <div class="node-label">${this._t('house')}</div>
                <div class="node-value">${this._formatPower(house)}</div>
              </div>

              <!-- Battery -->
              <div class="flow-node battery" style="--battery-color: ${batteryColor}">
                <ha-icon icon="mdi:battery${batteryStatus === 'charging' ? '-charging' : batteryStatus === 'discharging' ? '-arrow-down' : ''}"></ha-icon>
                <div class="node-label">${this._t('battery')}</div>
                <div class="node-value">${this._formatPower(batteryPower)}</div>
                <div class="battery-soc">
                  <div class="soc-bar">
                    <div class="soc-fill" style="width: ${batterySoc}%"></div>
                  </div>
                  <span>${this._formatPercent(batterySoc)}</span>
                </div>
              </div>

              <!-- Grid -->
              <div class="flow-node grid">
                <ha-icon icon="mdi:transmission-tower"></ha-icon>
                <div class="node-label">${this._t('grid')}</div>
                <div class="node-value-split">
                  <div class="grid-import">↓ ${this._formatPower(gridImport)}</div>
                  <div class="grid-export">↑ ${this._formatPower(gridExport)}</div>
                </div>
              </div>
            </div>
          </div>

          ${this._config.show_energy_flows ? this._renderEnergyFlows(
            solarToHouse, solarToBattery, solarToGrid,
            batteryToHouse, gridToHouse, gridToBattery
          ) : ''}

          ${this._renderAdditionalInfo()}
        </div>
      </ha-card>
    `;
  }

  /**
   * Render energy flows section - Simple List View
   */
  _renderEnergyFlows(solarToHouse, solarToBattery, solarToGrid, batteryToHouse, gridToHouse, gridToBattery) {
    // Get current live power values
    const solar = this._getState(this._entities.solarProduction) || 0;
    const batteryPower = this._getState(this._entities.batteryPower) || 0;
    const gridImport = this._getState(this._entities.gridImport) || 0;
    const gridExport = this._getState(this._entities.gridExport) || 0;
    const house = this._getState(this._entities.houseConsumption) || 0;

    // Calculate live flows
    const liveSolarToHouse = solar > 0 && house > 0 ? Math.min(solar, house) : 0;
    const liveSolarToBattery = solar > 0 && batteryPower > 0 ? batteryPower : 0;
    const liveSolarToGrid = solar > 0 && gridExport > 0 ? gridExport : 0;
    const liveBatteryToHouse = batteryPower < 0 ? Math.abs(batteryPower) : 0;
    const liveGridToHouse = gridImport > 0 ? gridImport : 0;
    const liveGridToBattery = gridImport > 0 && batteryPower > 0 ? Math.min(gridImport, batteryPower) : 0;

    const renderFlow = (from, to, liveW, totalKwh, color, icon) => `
      <div class="simple-flow">
        <div class="flow-from">
          <ha-icon icon="${icon}" style="color: ${color}"></ha-icon>
          <span>${from}</span>
        </div>
        <div class="flow-arrow" style="color: ${color}">→</div>
        <div class="flow-to">
          <span>${to}</span>
        </div>
        <div class="flow-values">
          <div class="live-w" style="color: ${liveW > 10 ? color : 'var(--secondary-text-color)'}">${this._formatPower(liveW)}</div>
          <div class="total-kwh">${this._formatEnergy(totalKwh)}</div>
        </div>
      </div>
    `;

    return `
      <div class="energy-flows-simple">
        <div class="section-title">
          <ha-icon icon="mdi:chart-timeline-variant"></ha-icon>
          <span>${this._t('energy_flows')}</span>
        </div>
        <div class="flows-list">
          ${renderFlow('Solar', 'Haus', liveSolarToHouse, solarToHouse, '#FF9800', 'mdi:solar-power')}
          ${renderFlow('Solar', 'Akku', liveSolarToBattery, solarToBattery, '#4CAF50', 'mdi:solar-power')}
          ${renderFlow('Solar', 'Netz', liveSolarToGrid, solarToGrid, '#FFC107', 'mdi:solar-power')}
          ${renderFlow('Akku', 'Haus', liveBatteryToHouse, batteryToHouse, '#8BC34A', 'mdi:battery-arrow-down')}
          ${renderFlow('Netz', 'Haus', liveGridToHouse, gridToHouse, '#2196F3', 'mdi:transmission-tower-import')}
          ${renderFlow('Netz', 'Akku', liveGridToBattery, gridToBattery, '#03A9F4', 'mdi:transmission-tower-import')}
        </div>
      </div>
    `;
  }

  /**
   * Render additional info (forecast, price, etc.)
   */
  _renderAdditionalInfo() {
    const runtime = this._getStateObj(this._entities.batteryRuntime);
    const efficiency = this._getState(this._entities.batteryEfficiency);
    const forecast = this._getState(this._entities.forecastTomorrow);
    const price = this._getState(this._entities.electricityPrice);

    let html = '<div class="additional-info">';

    if (runtime) {
      html += `
        <div class="info-item">
          <ha-icon icon="mdi:timer-sand"></ha-icon>
          <span>${this._t('runtime')}: ${runtime.state}</span>
        </div>
      `;
    }

    if (efficiency !== null) {
      html += `
        <div class="info-item">
          <ha-icon icon="mdi:chart-line"></ha-icon>
          <span>${this._t('efficiency')}: ${this._formatPercent(efficiency)}</span>
        </div>
      `;
    }

    if (this._config.show_forecast && forecast !== null) {
      html += `
        <div class="info-item">
          <ha-icon icon="mdi:weather-sunny"></ha-icon>
          <span>${this._t('forecast_tomorrow')}: ${this._formatEnergy(forecast)}</span>
        </div>
      `;
    }

    if (this._config.show_price && price !== null) {
      html += `
        <div class="info-item">
          <ha-icon icon="mdi:currency-eur"></ha-icon>
          <span>${this._t('price')}: ${price.toFixed(2)} ct/kWh</span>
        </div>
      `;
    }

    html += '</div>';
    return html;
  }

  /**
   * Get CSS styles
   */
  _getStyles() {
    return `
      :host {
        display: block;
      }

      ha-card {
        padding: 16px;
      }

      .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
      }

      .name {
        font-size: 24px;
        font-weight: 500;
        color: var(--primary-text-color);
      }

      .version {
        font-size: 12px;
        color: var(--secondary-text-color);
        background: var(--primary-color);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
      }

      .card-content {
        display: flex;
        flex-direction: column;
        gap: 24px;
      }

      /* Power Flow Diagram */
      .power-flow {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        padding: 16px;
        background: var(--card-background-color);
        border-radius: 8px;
      }

      .flow-node {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 12px 20px;
        background: var(--primary-background-color);
        border-radius: 12px;
        border: 2px solid var(--divider-color);
        min-width: 120px;
        transition: all 0.3s ease;
      }

      .flow-node:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      }

      .flow-node ha-icon {
        --mdi-icon-size: 32px;
        color: var(--primary-color);
        margin-bottom: 8px;
      }

      .flow-node.solar ha-icon {
        color: #FF9800;
      }

      .flow-node.battery ha-icon {
        color: var(--battery-color, #4CAF50);
      }

      .flow-node.grid ha-icon {
        color: #2196F3;
      }

      .flow-node.house ha-icon {
        color: #9E9E9E;
      }

      .flow-node.inverter ha-icon {
        color: #FFC107;
      }

      .node-label {
        font-size: 12px;
        color: var(--secondary-text-color);
        margin-bottom: 4px;
      }

      .node-value {
        font-size: 18px;
        font-weight: 600;
        color: var(--primary-text-color);
      }

      .node-sublabel {
        font-size: 10px;
        color: var(--secondary-text-color);
        margin-top: 2px;
      }

      /* Flow Arrows */
      .flow-arrow {
        position: relative;
        width: 4px;
        height: 40px;
        background: var(--divider-color);
        border-radius: 2px;
        transition: background 0.3s ease;
      }

      .flow-arrow.active {
        background: var(--primary-color);
        box-shadow: 0 0 10px var(--primary-color);
      }

      .flow-arrow.active::after {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 8px solid var(--primary-color);
      }

      .flow-split {
        width: 100%;
        height: 40px;
        position: relative;
      }

      .flow-arrow-split {
        width: 100%;
        height: 2px;
        background: var(--divider-color);
        position: relative;
        top: 50%;
      }

      .flow-arrow-split.active {
        background: var(--primary-color);
      }

      /* Destinations */
      .flow-destinations {
        display: flex;
        justify-content: space-around;
        width: 100%;
        gap: 12px;
        flex-wrap: wrap;
      }

      .flow-destinations .flow-node {
        flex: 1;
        min-width: 100px;
      }

      /* Battery SOC */
      .battery-soc {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 8px;
        width: 100%;
      }

      .soc-bar {
        flex: 1;
        height: 8px;
        background: var(--divider-color);
        border-radius: 4px;
        overflow: hidden;
      }

      .soc-fill {
        height: 100%;
        background: var(--battery-color, #4CAF50);
        transition: width 0.5s ease;
      }

      .battery-soc span {
        font-size: 12px;
        font-weight: 600;
        color: var(--primary-text-color);
      }

      /* Grid Import/Export */
      .node-value-split {
        display: flex;
        flex-direction: column;
        gap: 4px;
        font-size: 14px;
      }

      .grid-import {
        color: #F44336;
      }

      .grid-export {
        color: #4CAF50;
      }

      /* Energy Flow - Simple Horizontal List */
      .energy-flows-simple {
        margin-top: 24px;
      }

      .section-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 16px;
        font-weight: 500;
        color: var(--primary-text-color);
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--divider-color);
      }

      .section-title ha-icon {
        --mdi-icon-size: 22px;
        color: var(--primary-color);
      }

      .flows-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .simple-flow {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 18px;
        background: var(--card-background-color);
        border-radius: 12px;
        border: 1px solid var(--divider-color);
        transition: all 0.2s ease;
      }

      .simple-flow:hover {
        border-color: var(--primary-color);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }

      .flow-from {
        display: flex;
        align-items: center;
        gap: 10px;
        min-width: 120px;
      }

      .flow-from ha-icon {
        --mdi-icon-size: 24px;
        flex-shrink: 0;
      }

      .flow-from span {
        font-size: 14px;
        font-weight: 600;
        color: var(--primary-text-color);
        white-space: nowrap;
      }

      .flow-arrow {
        font-size: 20px;
        font-weight: 700;
        margin: 0 12px;
        flex-shrink: 0;
      }

      .flow-to {
        min-width: 80px;
      }

      .flow-to span {
        font-size: 14px;
        font-weight: 500;
        color: var(--secondary-text-color);
        white-space: nowrap;
      }

      .flow-values {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-left: auto;
        padding-left: 20px;
      }

      .live-w {
        font-size: 16px;
        font-weight: 700;
        min-width: 80px;
        text-align: right;
      }

      .total-kwh {
        font-size: 13px;
        color: var(--secondary-text-color);
        min-width: 70px;
        text-align: right;
      }

      @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.8); }
      }

      /* Additional Info */
      .additional-info {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }

      .info-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: var(--primary-background-color);
        border-radius: 8px;
        font-size: 14px;
        flex: 1;
        min-width: 150px;
      }

      .info-item ha-icon {
        --mdi-icon-size: 20px;
        color: var(--primary-color);
      }

      /* Responsive */
      @media (max-width: 768px) {
        .flow-destinations {
          flex-direction: column;
        }

        .simple-flow {
          padding: 12px 14px;
        }

        .flow-values {
          gap: 12px;
          padding-left: 12px;
        }

        .live-w {
          font-size: 15px;
          min-width: 70px;
        }

        .total-kwh {
          font-size: 12px;
          min-width: 60px;
        }
      }

      @media (max-width: 480px) {
        .simple-flow {
          flex-wrap: wrap;
          gap: 8px;
          padding: 10px 12px;
        }

        .flow-from {
          min-width: auto;
        }

        .flow-from span {
          font-size: 13px;
        }

        .flow-to {
          min-width: auto;
        }

        .flow-to span {
          font-size: 13px;
        }

        .flow-arrow {
          font-size: 18px;
          margin: 0 8px;
        }

        .flow-values {
          width: 100%;
          justify-content: flex-end;
          padding-left: 0;
          gap: 10px;
        }

        .live-w {
          font-size: 14px;
          min-width: 60px;
        }

        .total-kwh {
          font-size: 11px;
          min-width: 50px;
        }
      }
    `;
  }

  /**
   * Get card size for layout
   */
  getCardSize() {
    return 6;
  }

  /**
   * Get stub config for card picker
   */
  static getStubConfig() {
    return {
      type: 'custom:solar-forecast-ml-card',
      title: 'Solar Forecast ML',
      show_solar: true,
      show_battery: true,
      show_grid: true,
      show_energy_flows: true,
      show_forecast: true,
      show_price: true,
    };
  }
}

// Register the card
customElements.define('solar-forecast-ml-card', SolarForecastMLCard);

// Register with card picker
window.customCards = window.customCards || [];
window.customCards.push({
  type: 'solar-forecast-ml-card',
  name: 'Solar Forecast ML Energy Flow',
  description: 'v9.0.0 Battery Management Energy Flow Visualization',
  preview: true,
  documentationURL: 'https://github.com/Zara-Toorox/solar-forecast-ml',
});

console.info(
  '%c SOLAR-FORECAST-ML-CARD %c v9.0.0 ',
  'color: white; background: #FF9800; font-weight: 700;',
  'color: #FF9800; background: white; font-weight: 700;'
);
