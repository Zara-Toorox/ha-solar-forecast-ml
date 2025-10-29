"""
Scheduled Tasks Manager Module
Manages daily updates and verifications

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

Copyright (C) 2025 Zara-Toorox
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from homeassistant.core import HomeAssistant, callback

from .helpers import SafeDateTimeUtil as dt_util
# === START PATCH 3: Import für LearnedWeights ===
from .ml_types import LearnedWeights
# === ENDE PATCH 3 ===

_LOGGER = logging.getLogger(__name__)


class ScheduledTasksManager:
    """
    Manages scheduled tasks: Morning update and Evening verification.
    """
    
    def __init__(
        self,
        hass: HomeAssistant,
        coordinator,
        solar_yield_today: Optional[str],
        data_manager
    ):
        self.hass = hass
        self.coordinator = coordinator
        self.solar_yield_today = solar_yield_today
        self.data_manager = data_manager
    
    async def calculate_yesterday_deviation_on_startup(self) -> None:
        """
        Calculates the deviation from yesterday upon HA startup to populate
        the 'last_day_error_kwh' coordinator value.
        """
        _LOGGER.info("=== CALCULATING YESTERDAY'S DEVIATION ON STARTUP ===")
        
        try:
            history_data = await self.data_manager.get_prediction_history()
            predictions: List[Dict[str, Any]] = history_data.get('predictions', [])
            
            if not predictions:
                _LOGGER.info("No predictions available - setting deviation to 0")
                self.coordinator.last_day_error_kwh = 0.0
                return
            
            # Get yesterday's date in local timezone
            # *** KORREKTUR: 'dt_util.timedelta' zu 'timedelta' geändert ***
            yesterday = (dt_util.as_local(dt_util.utcnow()) - timedelta(days=1)).date()
            
            yesterday_predictions = []
            for pred in predictions:
                try:
                    # Parse stored timestamp (likely UTC) and convert to local date
                    timestamp = dt_util.parse_datetime(pred.get('timestamp', ''))
                    pred_date = dt_util.as_local(timestamp).date()
                    
                    if pred_date == yesterday:
                        yesterday_predictions.append(pred)
                except (ValueError, KeyError):
                    continue
            
            if not yesterday_predictions:
                _LOGGER.info(f"No predictions found for yesterday ({yesterday})")
                self.coordinator.last_day_error_kwh = 0.0
                return

            # --- CRITICAL BUG FIX ---
            # We must NOT sum up all predictions. We must take the *last*
            # prediction from yesterday, which contains the final forecast
            # and the actual value stored during evening verification.
            
            last_prediction = yesterday_predictions[-1]
            
            predicted_kwh = last_prediction.get('predicted_value', 0.0)
            actual_kwh = last_prediction.get('actual_value')
            
            if actual_kwh is None:
                _LOGGER.info(f"No actual_value available for yesterday - setting deviation to 0")
                self.coordinator.last_day_error_kwh = 0.0
                return
            
            deviation = abs(predicted_kwh - actual_kwh)
            self.coordinator.last_day_error_kwh = deviation
            
            _LOGGER.info(
                f"YESTERDAY'S DEVIATION CALCULATED:\n"
                f"   Date:       {yesterday}\n"
                f"   Predicted:  {predicted_kwh:.2f} kWh\n"
                f"   Actual:     {actual_kwh:.2f} kWh\n"
                f"   Deviation:  {deviation:.2f} kWh"
            )
            
        except Exception as e:
            _LOGGER.error(f"Error calculating yesterday's deviation: {e}", exc_info=True)
            self.coordinator.last_day_error_kwh = 0.0
    
    @callback
    async def scheduled_morning_update(self, now: datetime) -> None:
        """
        Scheduled task to run the daily morning forecast update.
        """
        _LOGGER.info("=== DAILY MORNING UPDATE STARTED ===")
        _LOGGER.info(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            await self.coordinator.async_request_refresh()
            _LOGGER.info("Daily forecast created successfully")
            
        except Exception as e:
            _LOGGER.error(f"Morning update failed: {e}", exc_info=True)
    
    @callback
    async def scheduled_evening_verification(self, now: datetime) -> None:
        """
        Scheduled task to verify today's forecast against the actual yield sensor.
        """
        _LOGGER.info("=== DAILY EVENING VERIFICATION STARTED ===")
        _LOGGER.info(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.solar_yield_today:
            _LOGGER.warning("No solar_yield_today sensor configured - skipping verification")
            return
        
        try:
            state = self.hass.states.get(self.solar_yield_today)
            
            if not state or state.state in ['unavailable', 'unknown', 'none', None]:
                _LOGGER.warning(f"Sensor {self.solar_yield_today} not available")
                return
            
            try:
                actual_kwh = float(state.state)
            except (ValueError, TypeError):
                _LOGGER.warning(f"Invalid value from {self.solar_yield_today}: {state.state}")
                return
            
            if not self.coordinator.data:
                _LOGGER.warning("No coordinator data available for verification")
                return
                
            predicted_kwh = self.coordinator.data.get("forecast_today", 0.0)
            
            if predicted_kwh > 0.1:
                error = abs(predicted_kwh - actual_kwh)
                relative_error = error / predicted_kwh
                accuracy = max(0.0, 1.0 - relative_error)
                
                # === START PATCH 3: FALLBACK-OPTIMIERUNG (SPEICHERN) ===
                if actual_kwh > 0.1: # Nur lernen, wenn auch Ertrag da war
                    try:
                        new_correction_factor = actual_kwh / predicted_kwh
                        # Begrenze den Faktor (Clamp 0.5 - 1.5)
                        new_correction_factor = max(0.5, min(1.5, new_correction_factor))
                        
                        _LOGGER.info(f"Calculating new fallback correction_factor: {actual_kwh:.2f} / {predicted_kwh:.2f} = {new_correction_factor:.3f}")
                        
                        # Lade die aktuellen Gewichte
                        current_weights = await self.data_manager.get_learned_weights()
                        
                        if current_weights:
                            current_weights.correction_factor = new_correction_factor
                            await self.data_manager.save_learned_weights(current_weights)
                            _LOGGER.info(f"Fallback correction_factor saved: {new_correction_factor:.3f}")
                        else:
                            # Falls noch kein ML-Modell trainiert wurde, erstellen wir Standardgewichte
                            # (Wir nehmen an, create_default_learned_weights existiert, basierend auf ml_predictor.py)
                            from .ml_types import create_default_learned_weights
                            _LOGGER.warning("No learned_weights.json found. Creating default weights to save correction_factor.")
                            default_weights = create_default_learned_weights()
                            default_weights.correction_factor = new_correction_factor
                            await self.data_manager.save_learned_weights(default_weights)
                            _LOGGER.info(f"Fallback correction_factor saved in new default weights file: {new_correction_factor:.3f}")
                            
                    except Exception as e:
                        _LOGGER.error(f"Failed to update fallback correction_factor: {e}", exc_info=True)
                # === ENDE PATCH 3 ===
                
            else:
                accuracy = 0.0  # Avoid division by zero, meaningless accuracy
            
            try:
                # Store the actual value in all prediction records made today
                await self.data_manager.update_today_predictions_actual(actual_kwh, accuracy)
                _LOGGER.debug("All of today's prediction records updated with actual_value")
            except Exception as e:
                _LOGGER.warning(f"Could not update today's prediction records: {e}")
            
            _LOGGER.info(
                f"   DAILY FORECAST CHECK:\n"
                f"   Predicted: {predicted_kwh:.2f} kWh\n"
                f"   Actual:    {actual_kwh:.2f} kWh\n"
                f"   Error:     {abs(predicted_kwh - actual_kwh):.2f} kWh\n"
                f"   Accuracy:  {accuracy*100:.1f}%"
            )
            
            # Update the coordinator state for tomorrow
            self.coordinator.last_day_error_kwh = abs(predicted_kwh - actual_kwh)
            
        except Exception as e:
            _LOGGER.error(f"Evening verification failed: {e}", exc_info=True)