"""
Data Update Coordinator für Solar Forecast ML Integration.
✓ ADVANCED VERSION: Vollständige Strategy-Integration
✓ Nutzt ML & Rule-Based Strategies komplett
✓ FIX: Korrektes Data-Mapping für Sensoren # von Zara
✓ STRATEGIE 2: Zentrale Properties und Produktionszeit-Tracking # von Zara
✓ FIX: Verbesserter Produktionszeit-Fallback # von Zara
✓ SONNENSTAND: Next Hour Prediction basierend auf sun.sun Entity # von Zara
Version 4.12.0 - Sonnenstand-basierte Next Hour Prediction # von Zara

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2025 Zara-Toorox
# von Zara
"""
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.helpers.event import async_track_state_change_event, async_track_time_change
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN, CONF_SOLAR_CAPACITY, CONF_WEATHER_ENTITY,
    CONF_LEARNING_ENABLED, UPDATE_INTERVAL, CONF_PLANT_KWP,
    CONF_HOURLY, CONF_POWER_ENTITY, CONF_SOLAR_YIELD_TODAY,
    CORRECTION_FACTOR_MIN, CORRECTION_FACTOR_MAX, DATA_DIR
)
from .data_manager import DataManager
from .exceptions import SolarForecastMLException, WeatherException, ModelException

# Importiere Module (Flache Struktur) # von Zara
from .ml_forecast_strategy import MLForecastStrategy
from .rule_based_forecast_strategy import RuleBasedForecastStrategy
from .service_manager import ServiceManager
from .weather_calculator import WeatherCalculator
from .production_calculator import ProductionCalculator, ProductionTimeCalculator

_LOGGER = logging.getLogger(__name__)


class SolarForecastMLCoordinator(DataUpdateCoordinator):
    """
    Coordinator für Solar Forecast ML Integration.
    ✓ ADVANCED: Vollständige Strategy-Integration
    ✓ SMART: Automatische Fallback zwischen ML und Rule-Based
    ✓ STRATEGIE 2: Zentrale Properties für alle Sensoren # von Zara
    # von Zara
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        dependencies_ok: bool = False,  # 🔧 NEU: Dependencies-Status # von Zara
    ):
        """
        Initialize the coordinator.
        
        Args:
            hass: HomeAssistant Instanz
            entry: ConfigEntry
            dependencies_ok: True wenn alle Dependencies vorhanden # von Zara
        # von Zara
        """
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=UPDATE_INTERVAL,
        )
        
        self.entry = entry
        
        # ✓ DataManager mit entry_id und data_dir initialisieren # von Zara
        self.data_manager = DataManager(hass, entry.entry_id, Path(DATA_DIR))
        
        # Configuration # von Zara
        self.solar_capacity = entry.data.get(CONF_SOLAR_CAPACITY, 5.0)
        self.learning_enabled = entry.data.get(CONF_LEARNING_ENABLED, True)
        self.power_entity = entry.data.get(CONF_POWER_ENTITY)
        self.solar_yield_today = entry.data.get(CONF_SOLAR_YIELD_TODAY)  # Tagesertrag Sensor - von Zara
        
        # Weather Entity aus Config # von Zara
        self.primary_weather_entity = entry.data.get(CONF_WEATHER_ENTITY)
        self.current_weather_entity: Optional[str] = self.primary_weather_entity
        
        # Tracking # von Zara
        self.weather_fallback_active = False
        self.enable_hourly = entry.data.get(CONF_HOURLY, entry.options.get(CONF_HOURLY, False))  # Fix: Priorität auf entry.data - von Zara
        
        # Status tracking # von Zara
        self._last_weather_update = None
        self._forecast_cache = {}
        
        # ✓ STRATEGIE 2: Zentrale Timestamp-Tracking # von Zara
        self._last_update_success_time: Optional[datetime] = None
        self.last_update_time: Optional[datetime] = None  # Für Sensoren - von Zara
        
        # Sensor-kompatible Properties # von Zara
        self.next_hour_pred = 0.0
        self.peak_production_time_today = "12:00"
        self.production_time_today = "Initialisierung..."
        self.last_day_error_kwh = None
        self.autarky_today = None
        self.average_yield_30_days = 0.0
        self.last_successful_learning = None
        self.model_accuracy = None
        
        # ✓ Initialisiere Calculators # von Zara
        self.weather_calculator = WeatherCalculator()
        self.production_calculator = ProductionCalculator(hass)
        
        # ✓ STRATEGIE 2: Produktionszeit-Tracking # von Zara
        self.production_time_calculator = ProductionTimeCalculator(
            hass=hass,
            power_entity=self.power_entity
        )
        
        # ✓ Service Manager für alle Services # von Zara
        self.service_manager = ServiceManager(
            hass=hass,
            entry=entry,
            data_manager=self.data_manager,
            weather_entity=self.current_weather_entity,
            dependencies_ok=dependencies_ok  # 🔧 FIX: Dependencies-Status weitergeben # von Zara
        )
        
        # ✓ ADVANCED: Forecast Strategies # von Zara
        self.ml_strategy: Optional[MLForecastStrategy] = None
        self.rule_based_strategy: Optional[RuleBasedForecastStrategy] = None
        self.active_strategy: Optional[str] = None
        
        
        _LOGGER.info(
            f"✓ SolarForecastMLCoordinator (STRATEGIE 2) initialisiert - "
            f"Weather Entity: {self.primary_weather_entity}"
        )
    
    async def async_config_entry_first_refresh(self) -> None:
        """
        ✓ Override: First Refresh mit Weather Entity Wartezeit
        ✓ ADVANCED: Initialisiere Strategies
        ✓ STRATEGIE 2: Starte Produktionszeit-Tracking # von Zara
        # von Zara
        """
        _LOGGER.info("=== Starting First Refresh (STRATEGIE 2) ===")
        
        # Setze Weather Entity # von Zara
        self.current_weather_entity = self.primary_weather_entity
        _LOGGER.info(f"✓ Weather Entity konfiguriert: {self.current_weather_entity}")
        
        # Warte auf Weather Entity Verfügbarkeit # von Zara
        if self.current_weather_entity:
            _LOGGER.info("⏳ Warte auf Weather Entity Verfügbarkeit...")
            max_wait_time = 30
            wait_interval = 2
            total_waited = 0
            
            while total_waited < max_wait_time:
                if await self._check_weather_entity_available(self.current_weather_entity):
                    _LOGGER.info(
                        f"✓ Weather Entity '{self.current_weather_entity}' ist bereit "
                        f"(nach {total_waited}s)"
                    )
                    break
                
                await asyncio.sleep(wait_interval)
                total_waited += wait_interval
            
            if total_waited >= max_wait_time:
                _LOGGER.warning(
                    f"⚠️ Weather Entity '{self.current_weather_entity}' nicht verfügbar "
                    f"nach {max_wait_time}s - starte trotzdem"
                )
        
        # ✓ ADVANCED: Initialisiere Strategies # von Zara
        try:
            # Rule-Based Strategy ist immer verfügbar # von Zara
            self.rule_based_strategy = RuleBasedForecastStrategy(
                solar_capacity=self.solar_capacity,
                weather_calculator=self.weather_calculator
            )
            _LOGGER.info("✓ Rule-Based Strategy initialisiert")
            
            # ML Strategy wenn ML Predictor verfügbar # von Zara
            ml_predictor = self.service_manager.ml_predictor
            if ml_predictor:
                self.ml_strategy = MLForecastStrategy(
                    ml_predictor=ml_predictor,
                    error_handler=self.service_manager.error_handler
                )
                _LOGGER.info("✓ ML Strategy initialisiert")
            else:
                _LOGGER.info("ℹ️ ML Strategy nicht verfügbar (Predictor fehlt)")
                
        except Exception as e:
            _LOGGER.error(f"❌ Strategy Initialisierung fehlgeschlagen: {e}")
        
        # ✓ STRATEGIE 2: Starte Produktionszeit-Tracking # von Zara
        try:
            if self.power_entity:
                self.production_time_calculator.start_tracking()
                _LOGGER.info(f"✓ Produktionszeit-Tracking gestartet für {self.power_entity}")
            else:
                _LOGGER.info("ℹ️ Kein Power-Sensor konfiguriert - Produktionszeit-Tracking deaktiviert")
        except Exception as e:
            _LOGGER.warning(f"⚠️ Produktionszeit-Tracking konnte nicht gestartet werden: {e}")
        
        # Standard first refresh # von Zara
        await super().async_config_entry_first_refresh()
    
    async def _check_weather_entity_available(self, entity_id: str) -> bool:
        """
        Prüft ob Weather Entity verfügbar ist.
        
        Args:
            entity_id: Entity ID zu prüfen
            
        Returns:
            True wenn verfügbar # von Zara
        """
        if not entity_id:
            return False
        
        state = self.hass.states.get(entity_id)
        if state is None:
            return False
        
        if state.state in ["unavailable", "unknown"]:
            return False
        
        return True
    
    async def _get_weather_data(self) -> Dict[str, Any]:
        """
        Hole Wetterdaten vom konfigurierten Weather Entity.
        
        Returns:
            Dictionary mit Wetterdaten
            
        Raises:
            WeatherException bei Fehlern # von Zara
        """
        if not self.current_weather_entity:
            raise WeatherException("Keine Weather Entity konfiguriert")
        
        state = self.hass.states.get(self.current_weather_entity)
        
        if state is None:
            raise WeatherException(f"Weather Entity {self.current_weather_entity} nicht verfügbar")
        
        if state.state in ["unavailable", "unknown"]:
            raise WeatherException(
                f"Weather Entity {self.current_weather_entity} ist {state.state}"
            )
        
        # Extrahiere Wetterdaten aus State Attributes # von Zara
        try:
            attributes = state.attributes
            
            # Cloud Cover aus Condition ableiten # von Zara
            condition = state.state.lower()
            cloud_cover_map = {
                "clear-night": 0,
                "sunny": 0,
                "partlycloudy": 40,
                "cloudy": 80,
                "rainy": 90,
                "snowy": 90,
                "fog": 100
            }
            cloud_cover = cloud_cover_map.get(condition, 50)
            
            weather_data = {
                "temperature": float(attributes.get("temperature", 15.0)),
                "humidity": float(attributes.get("humidity", 60.0)),
                "cloud_cover": float(cloud_cover),
                "wind_speed": float(attributes.get("wind_speed", 3.0)),
                "precipitation": float(attributes.get("precipitation", 0.0)),
                "pressure": float(attributes.get("pressure", 1013.25)),
                "condition": condition
            }
            
            _LOGGER.debug(
                f"✓ Weather Data: Temp={weather_data['temperature']}°C, "
                f"Clouds={weather_data['cloud_cover']}%, "
                f"Condition={weather_data['condition']}"
            )
            
            self._last_weather_update = dt_util.utcnow()
            
            return weather_data
            
        except (ValueError, TypeError, KeyError) as e:
            raise WeatherException(f"Fehler beim Parsen der Wetterdaten: {e}")
    
    async def _async_update_data(self) -> dict[str, Any]:
        """
        ✓ Update Daten und erstelle Forecasts mit Strategy
        # von Zara
        """
        try:
            _LOGGER.debug("=== Starting Data Update (STRATEGIE 2) ===")
            
            # Lade Wetterdaten # von Zara
            weather_data = await self._get_weather_data()
            _LOGGER.debug(f"Weather Data: {weather_data}")
            
            # Erstelle Forecast mit Strategy # von Zara
            forecast = await self._create_forecast_with_strategy(weather_data)
            
            
            # Berechne historische Peak-Zeit aus Power-Daten # von Zara
            try:
                if self.power_entity:
                    historical_peak = await self.production_calculator.calculate_peak_production_time(
                        power_entity=self.power_entity
                    )
                    # Überschreibe nur wenn Default-Wert # von Zara
                    if forecast.get("peak_time") == "12:00" or not forecast.get("peak_time"):
                        forecast["peak_time"] = historical_peak
                        _LOGGER.debug(f"✓ Historische Peak-Zeit verwendet: {historical_peak}")
                    else:
                        _LOGGER.debug(f"ℹ️ ML Peak-Zeit beibehalten: {forecast.get('peak_time')}")
                else:
                    _LOGGER.debug("ℹ️ Kein Power-Sensor - verwende Standard Peak-Zeit")
            except Exception as e:
                _LOGGER.warning(f"⚠️ Peak-Zeit Berechnung fehlgeschlagen: {e}")
            
            # Update Sensor Properties # von Zara
            self._update_sensor_properties(forecast)
            
            # Speichere Success Time # von Zara
            self._last_update_success_time = dt_util.now()
            
            # ✓ ZUSÄTZLICHE PROPERTY-UPDATES - von Zara
            self.last_update_time = dt_util.utcnow()  # Für Sensoren - von Zara
            
            # ✅ Next Hour Prediction mit Sonnenstand berechnen - von Zara
            try:
                # Hole sun.sun Entity für präzise Sonnenstand-Berechnung - von Zara
                sun_state = self.hass.states.get("sun.sun")
                
                if sun_state and sun_state.state not in ['unavailable', 'unknown']:
                    # Prüfe Sonnenhöhe (elevation in Grad: -90° bis 90°) - von Zara
                    elevation = sun_state.attributes.get("elevation", 0)
                    
                    if elevation <= 0:
                        # Sonne unter Horizont → keine Produktion - von Zara
                        self.next_hour_pred = 0.0
                        _LOGGER.debug(f"🌙 Sonne unter Horizont (elevation={elevation}°) → next_hour_pred=0.0")
                    else:
                        # Sonne über Horizont → proportionale Berechnung - von Zara
                        # Peak bei ~60° → Faktor 1.0, niedrig bei 5° → Faktor ~0.08 - von Zara
                        sun_factor = min(elevation / 60.0, 1.0)
                        
                        # Verteile Tagesprognose auf ~15 produktive Sonnenstunden - von Zara
                        hourly_base = forecast.get("today", 0.0) / 15.0
                        self.next_hour_pred = round(hourly_base * sun_factor, 2)
                        
                        _LOGGER.debug(
                            f"☀️ Sonnenstand-Berechnung: elevation={elevation}°, "
                            f"sun_factor={sun_factor:.2f}, next_hour_pred={self.next_hour_pred} kWh"
                        )
                else:
                    # Fallback: Einfache Zeit-basierte Prüfung wenn sun.sun nicht verfügbar - von Zara
                    now = dt_util.utcnow()
                    if 21 <= now.hour or now.hour <= 5:
                        # Nachtstunden: keine Produktion - von Zara
                        self.next_hour_pred = 0.0
                        _LOGGER.debug(f"🌙 Nacht-Fallback (Stunde={now.hour}) → next_hour_pred=0.0")
                    else:
                        # Tagsüber: einfache Verteilung - von Zara
                        self.next_hour_pred = round(forecast.get("today", 0.0) / 15.0, 2)
                        _LOGGER.debug(f"☀️ Tag-Fallback → next_hour_pred={self.next_hour_pred} kWh")
                        
            except Exception as e:
                # Ultimate Fallback bei jedem Fehler - von Zara
                _LOGGER.debug(f"⚠️ Next Hour Berechnung fehlgeschlagen: {e}")
                self.next_hour_pred = 0.0
            
            # ✓ ML Training Status synchronisieren - von Zara
            try:
                ml_predictor = self.ml_predictor
                if ml_predictor and hasattr(ml_predictor, 'last_training_time'):
                    if ml_predictor.last_training_time:
                        self.last_successful_learning = ml_predictor.last_training_time
            except Exception as e:
                _LOGGER.debug(f"ML Status Sync fehlgeschlagen: {e}")
            
            # ✓ JSON Auto-Save triggern - von Zara
            try:
                if hasattr(self, 'data_manager') and self.data_manager:
                    await self.data_manager.save_all_async()
            except Exception as e:
                _LOGGER.debug(f"Auto-Save fehlgeschlagen: {e}")
            
            result = {
                "forecast_today": forecast["today"],
                "forecast_tomorrow": forecast["tomorrow"],
                "peak_time": forecast.get("peak_time", "12:00"),
                "confidence": forecast.get("confidence", 75.0),
                "method": forecast.get("method", "unknown"),
                "last_update": dt_util.utcnow().isoformat()
            }
            
            _LOGGER.debug(f"✓ Update complete: {result}")
            
            return result
            
        except WeatherException as e:
            _LOGGER.error(f"❌ Weather Error: {e}")
            raise UpdateFailed(f"Weather Fehler: {e}")
        except Exception as e:
            _LOGGER.error(f"❌ Update Failed: {e}", exc_info=True)
            raise UpdateFailed(f"Update Fehler: {e}")
    
    async def _create_forecast_with_strategy(
        self,
        weather_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Erstellt Forecast mit verfügbarer Strategy.
        
        Args:
            weather_data: Wetterdaten
            
        Returns:
            Forecast Dictionary mit today, tomorrow, peak_time, confidence, method
        # von Zara
        """
        # Sensor-Daten sammeln # von Zara
        sensor_data = {
            'solar_capacity': self.solar_capacity,
            'power_entity': self.power_entity
        }
        
        # Optional sensors # von Zara
        from .const import (
            CONF_TEMP_SENSOR, CONF_WIND_SENSOR, CONF_RAIN_SENSOR,
            CONF_UV_SENSOR, CONF_LUX_SENSOR, CONF_CURRENT_POWER
        )
        
        optional_sensors = {
            'temp_sensor': CONF_TEMP_SENSOR,
            'wind_sensor': CONF_WIND_SENSOR,
            'rain_sensor': CONF_RAIN_SENSOR,
            'uv_sensor': CONF_UV_SENSOR,
            'lux_sensor': CONF_LUX_SENSOR,
            'current_power': CONF_CURRENT_POWER
        }
        
        for key, config_key in optional_sensors.items():
            entity_id = self.entry.data.get(config_key)
            if entity_id:
                state = self.hass.states.get(entity_id)
                if state and state.state not in ['unavailable', 'unknown']:
                    try:
                        sensor_data[key] = float(state.state)
                        _LOGGER.debug(f"✓ Sensor {key} geladen: {sensor_data[key]}")
                    except (ValueError, TypeError):
                        _LOGGER.debug(f"⚠️ Sensor {key} Wert konnte nicht konvertiert werden")
        
        correction_factor = 1.0
        
        # Versuche ML Strategy zuerst # von Zara
        if self.ml_strategy and self.ml_strategy.is_available():
            try:
                _LOGGER.debug("🧠 Verwende ML Strategy für Forecast")
                result = await self.ml_strategy.calculate_forecast(
                    weather_data=weather_data,
                    sensor_data=sensor_data,
                    correction_factor=correction_factor
                )
                
                self.active_strategy = "ml"
                self.model_accuracy = result.model_accuracy
                
                return {
                    "today": result.forecast_today,
                    "tomorrow": result.forecast_tomorrow,
                    "peak_time": "12:00",
                    "confidence": result.confidence_today,
                    "method": result.method
                }
                
            except Exception as e:
                _LOGGER.warning(f"⚠️ ML Strategy fehlgeschlagen: {e}, Fallback zu Rule-Based")
        
        # Fallback zu Rule-Based Strategy # von Zara
        if self.rule_based_strategy:
            _LOGGER.debug("📊 Verwende Rule-Based Strategy für Forecast")
            result = await self.rule_based_strategy.calculate_forecast(
                weather_data=weather_data,
                sensor_data=sensor_data,
                correction_factor=correction_factor
            )
            
            self.active_strategy = "rule_based"
            
            return {
                "today": result.forecast_today,
                "tomorrow": result.forecast_tomorrow,
                "peak_time": "12:00",
                "confidence": result.confidence_today,
                "method": result.method
            }
        
        # Wenn nichts funktioniert, einfache Berechnung # von Zara
        _LOGGER.warning("⚠️ Keine Strategy verfügbar, verwende einfache Berechnung")
        return await self._simple_forecast(weather_data)
    
    async def _simple_forecast(self, weather_data: dict[str, Any]) -> dict[str, Any]:
        """Einfache Fallback-Berechnung # von Zara"""
        cloud_factor = 1.0 - (weather_data["cloud_cover"] / 100.0)
        temp_factor = self._calculate_temperature_factor(weather_data["temperature"])
        
        base_production = self.solar_capacity * 4.0
        today_forecast = base_production * cloud_factor * temp_factor
        
        return {
            "today": round(today_forecast, 2),
            "tomorrow": round(today_forecast * 0.95, 2),
            "peak_time": "12:00",
            "confidence": round(85.0 * cloud_factor, 1),
            "method": "simple"
        }
    
    def _calculate_temperature_factor(self, temperature: float) -> float:
        """
        Berechnet Temperatur-Faktor für Forecast
        # von Zara
        """
        optimal_temp = 25.0
        temp_diff = abs(temperature - optimal_temp)
        factor = 1.0 - (temp_diff * 0.005)
        return max(0.7, min(1.0, factor))
    
    def _update_sensor_properties(self, data: Dict[str, Any]) -> None:
        """
        ✓ STRATEGIE 2: Update zentrale Properties für Sensoren
        ✓ FIX: Verbesserter Fallback für Produktionszeit # von Zara
        # von Zara
        """
        # Update Peak Time # von Zara
        if "peak_time" in data:
            self.peak_production_time_today = data["peak_time"]
        
        # ✓ FIX: Update Produktionszeit mit Fallback # von Zara
        try:
            production_time = self.production_time_calculator.get_production_time()
            
            # Prüfe ob valide Zeit zurückkam # von Zara
            if production_time and production_time not in ["Nicht verfügbar", "Fehler"]:
                self.production_time_today = production_time
            elif not self.power_entity:
                # Kein Power-Sensor konfiguriert # von Zara
                self.production_time_today = "Kein Power-Sensor"
            elif production_time == "0h 0m":
                # Noch keine Produktion heute # von Zara
                now = dt_util.utcnow()
                if 5 <= now.hour <= 21:
                    # Während Tageszeit - noch keine Produktion # von Zara
                    self.production_time_today = "Noch keine Produktion"
                else:
                    # Nachts - zeige 0h 0m # von Zara
                    self.production_time_today = "0h 0m"
            else:
                # Verwende was der Calculator zurückgab # von Zara
                self.production_time_today = production_time
                
        except Exception as e:
            _LOGGER.debug(f"Produktionszeit-Update fehlgeschlagen: {e}")
            self.production_time_today = "Berechnung läuft..."
    
    # ========================================================================
    # ✓ STRATEGIE 2: ZENTRALE PROPERTIES FÜR SENSOREN # von Zara
    # ========================================================================
    
    @property
    def last_update_success_time(self) -> Optional[datetime]:
        """✓ STRATEGIE 2: Zentrale Property für letztes erfolgreiches Update # von Zara"""
        return self._last_update_success_time
    
    @property
    def ml_predictor(self):
        """✓ STRATEGIE 2: Zentrale Property für ML Predictor Zugriff # von Zara"""
        if hasattr(self.service_manager, 'ml_predictor'):
            return self.service_manager.ml_predictor
        return None
    
    @property
    def weather_source(self) -> str:
        """✓ Liefert aktuellen Weather Source für Sensor-Anzeige # von Zara"""
        return self.current_weather_entity or "Nicht verfügbar"
    
    @property
    def retry_attempts(self) -> int:
        """✓ Liefert Anzahl Retry-Versuche für Sensor-Anzeige # von Zara"""
        return 0  # Keine Retry-Logik # von Zara
    
    # ========================================================================
    # ✓ STRATEGIE 2: CALLBACKS FÜR ML-EVENTS # von Zara
    # ========================================================================
    
    def on_ml_training_complete(self, timestamp: datetime, accuracy: float = None) -> None:
        """
        ✓ STRATEGIE 2: Callback wenn ML-Training abgeschlossen
        # von Zara
        """
        _LOGGER.info(f"✓ ML-Training abgeschlossen - Accuracy: {accuracy}")
        self.last_successful_learning = timestamp
        if accuracy is not None:
            self.model_accuracy = accuracy
