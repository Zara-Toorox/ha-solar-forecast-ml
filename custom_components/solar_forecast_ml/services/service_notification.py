"""Notification Service for Solar Forecast ML Integration V12.2.0 @zara

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
"""

import asyncio
import logging
from typing import List, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from ..const import (
    CONF_NOTIFY_FORECAST,
    CONF_NOTIFY_FROST,
    CONF_NOTIFY_LEARNING,
    CONF_NOTIFY_STARTUP,
    CONF_NOTIFY_SUCCESSFUL_LEARNING,
    CONF_NOTIFY_WEATHER_ALERT,
)

_LOGGER = logging.getLogger(__name__)

NOTIFICATION_ID_DEPENDENCIES = "solar_forecast_ml_dependencies"
NOTIFICATION_ID_INSTALLATION = "solar_forecast_ml_installation"
NOTIFICATION_ID_SUCCESS = "solar_forecast_ml_success"
NOTIFICATION_ID_ERROR = "solar_forecast_ml_error"
NOTIFICATION_ID_ML_ACTIVE = "solar_forecast_ml_ml_active"
NOTIFICATION_ID_STARTUP = "solar_forecast_ml_startup"
NOTIFICATION_ID_FORECAST = "solar_forecast_ml_forecast"
NOTIFICATION_ID_LEARNING = "solar_forecast_ml_learning"
NOTIFICATION_ID_RETRAINING = "solar_forecast_ml_retraining"
NOTIFICATION_ID_FROST = "solar_forecast_ml_frost"
NOTIFICATION_ID_WEATHER_ALERT = "solar_forecast_ml_weather_alert"

class NotificationService:
    """Service for Persistent Notifications in Home Assistant"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize Notification Service @zara"""
        self.hass = hass
        self.entry = entry
        self._initialized = False
        self._notification_lock = asyncio.Lock()
        _LOGGER.debug("NotificationService instance created")

    async def initialize(self) -> bool:
        """Initialize the Notification Service @zara"""
        try:
            async with self._notification_lock:
                if self._initialized:
                    _LOGGER.debug("[OK] NotificationService already initialized")
                    return True

                if "persistent_notification" not in self.hass.config.components:
                    _LOGGER.warning(
                        "[!] persistent_notification not available - "
                        "Notifications will not be displayed"
                    )
                    self._initialized = True
                    return False

                self._initialized = True
                _LOGGER.info("[OK] NotificationService successfully initialized")
                return True

        except Exception as e:
            _LOGGER.error(
                f"[X] Error during NotificationService initialization: {e}", exc_info=True
            )
            return False

    def _should_notify(self, notification_type: str) -> bool:
        """Centralized check if notification should be displayed @zara"""
        if not self._initialized:
            return False

        enabled = self.entry.options.get(notification_type, True)

        if not enabled:
            _LOGGER.debug(f"Notification '{notification_type}' disabled by option")

        return enabled

    async def _safe_create_notification(
        self, message: str, title: str, notification_id: str
    ) -> bool:
        """Create notification with error handling"""
        if not self._initialized:
            _LOGGER.warning(
                f"[!] NotificationService not initialized - "
                f"Notification '{notification_id}' will not be displayed"
            )
            return False

        try:
            await self.hass.services.async_call(
                domain="persistent_notification",
                service="create",
                service_data={
                    "message": message,
                    "title": title,
                    "notification_id": notification_id,
                },
                blocking=True,
            )
            _LOGGER.debug(f"[OK] Notification '{notification_id}' created")
            return True

        except Exception as e:
            _LOGGER.error(
                f"[X] Error creating notification '{notification_id}': {e}", exc_info=True
            )
            return False

    async def _safe_dismiss_notification(self, notification_id: str) -> bool:
        """Remove notification with error handling @zara"""
        if not self._initialized:
            return False

        try:
            await self.hass.services.async_call(
                domain="persistent_notification",
                service="dismiss",
                service_data={
                    "notification_id": notification_id,
                },
                blocking=True,
            )
            _LOGGER.debug(f"[OK] Notification '{notification_id}' dismissed")
            return True

        except Exception as e:
            _LOGGER.warning(f"[!] Error dismissing notification '{notification_id}': {e}")
            return False

    async def show_startup_success(
        self,
        ml_mode: bool = True,
        installed_packages: Optional[List[str]] = None,
        missing_packages: Optional[List[str]] = None,
    ) -> bool:
        """Show startup notification with integration status"""
        if not self._should_notify(CONF_NOTIFY_STARTUP):
            return False

        try:

            installed_list = ""
            if installed_packages:
                installed_items = "\n".join([f"✓ {pkg}" for pkg in installed_packages])
                installed_list = f"\n\n**Installed Dependencies:**\n{installed_items}"

            missing_list = ""
            if missing_packages:
                missing_items = "\n".join([f"✗ {pkg}" for pkg in missing_packages])
                missing_list = f"\n\n**Missing Packages:**\n{missing_items}"

            if ml_mode:
                message = f"""**Solar Forecast KI Started Successfully!** ⭐

**Mode:** Machine Learning (Full Feature Set)

**Version:** "Sarpeidon" - Named after the planet from Star Trek where the Guardian of Forever resides

**Author:** Zara-Toorox

**Active Features:**
• ML-based solar production forecasting (14 features, weather-corrected)
• Historical data analysis & learning
• Weather integration for accuracy
• Peak production time detection
• Autarky & self-sufficiency tracking
• Daily Solar Briefing notifications{installed_list}

**System Status:** All systems operational ✓

*"The future is not set in stone, but with data and logic, we can illuminate the path ahead."* — Inspired by Star Trek

**Personal Note from Zara:**
Thank you for using Solar Forecast ML! May your panels generate efficiently. Live long and prosper! 🖖"""
            else:
                message = f"""**Solar Forecast ML Started in Fallback Mode** ⚠️

**Mode:** Rule-Based Calculations (Limited Features)

**Version:** "Sarpeidon" - Named after the planet from Star Trek where the Guardian of Forever resides

**Author:** Zara-Toorox

**Active Features:**
• Rule-based solar forecasting
• Historical data tracking
• Basic production statistics{missing_list}{installed_list}

**Note:** Install missing Python packages to enable ML features. The integration will continue working with rule-based calculations.

*"Even in the absence of certainty, we must still chart our course."* — Inspired by Star Trek

**Personal Note from Zara:**
Thank you for using Solar Forecast ML! Install the missing dependencies to unlock the full power of machine learning. 🖖"""

            await self._safe_create_notification(
                message=message,
                title="🌤️ Solar Forecast ML Started",
                notification_id=NOTIFICATION_ID_STARTUP,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing startup notification: {e}", exc_info=True)
            return False

    async def show_forecast_update(
        self, forecast_energy: float, confidence: Optional[float] = None
    ) -> bool:
        """Show forecast update notification"""
        if not self._should_notify(CONF_NOTIFY_FORECAST):
            return False

        try:
            confidence_text = ""
            if confidence is not None:
                confidence_text = f"\n**Confidence:** {confidence:.1f}%"

            message = f"""Solar Forecast Updated"""

            await self._safe_create_notification(
                message=message, title="Forecast Updated", notification_id=NOTIFICATION_ID_FORECAST
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing forecast notification: {e}", exc_info=True)
            return False

    async def show_training_start(self, sample_count: int) -> bool:
        """Show notification when ML training starts @zara"""
        if not self._should_notify(CONF_NOTIFY_LEARNING):
            return False

        try:
            message = f"""ML Training Started"""

            await self._safe_create_notification(
                message=message, title="Training Started", notification_id=NOTIFICATION_ID_LEARNING
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing training start notification: {e}", exc_info=True)
            return False

    async def show_training_complete(
        self, success: bool, accuracy: Optional[float] = None, sample_count: Optional[int] = None
    ) -> bool:
        """Show notification when ML training completes"""
        if not self._should_notify(CONF_NOTIFY_SUCCESSFUL_LEARNING):
            return False

        try:
            if success:
                accuracy_text = ""
                if accuracy is not None:
                    accuracy_text = f"\n**Accuracy:** {accuracy:.1f}%"

                sample_text = ""
                if sample_count is not None:
                    sample_text = f"\n**Samples Used:** {sample_count}"

                message = f"""OK ML Training Complete"""
            else:
                message = """ ML Training Failed"""

            await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)

            await self._safe_create_notification(
                message=message, title="Training Complete", notification_id=NOTIFICATION_ID_LEARNING
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing training complete notification: {e}", exc_info=True)
            return False

    async def dismiss_startup_notification(self) -> bool:
        """Remove startup notification @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_STARTUP)

    async def dismiss_forecast_notification(self) -> bool:
        """Remove forecast notification @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_FORECAST)

    async def dismiss_training_notification(self) -> bool:
        """Remove training notification @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)

    async def show_model_retraining_required(
        self,
        reason: str = "unknown",
        old_features: Optional[int] = None,
        new_features: Optional[int] = None,
    ) -> bool:
        """Show notification when ML model needs retraining"""
        try:

            if reason == "feature_mismatch":
                reason_text = f"""**Grund:** Sensoränderung erkannt

**Details:**
• Alte Features: {old_features}
• Neue Features: {new_features}

Das ML-Modell wird automatisch neu trainiert, um die geänderte Sensorkonfiguration zu berücksichtigen."""
            else:
                reason_text = "Das ML-Modell muss neu trainiert werden."

            message = f"""**Solar Forecast ML - Modell-Neutraining erforderlich** ⚠️

{reason_text}

**Nächste Schritte:**
• Das Training wird automatisch durchgeführt
• Bei Bedarf manuell starten: Service `solar_forecast_ml.force_retrain`

**Status:** Automatisches Training läuft...

*"Anpassung ist der Schlüssel zum Überleben."* — Inspired by Star Trek

**Personal Note from Zara:**
Keine Sorge! Die Integration passt sich automatisch an. 🖖"""

            await self._safe_create_notification(
                message=message,
                title="🔄 ML-Modell Neutraining",
                notification_id=NOTIFICATION_ID_RETRAINING,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing retraining notification: {e}", exc_info=True)
            return False

    async def dismiss_retraining_notification(self) -> bool:
        """Remove retraining notification @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_RETRAINING)

    async def show_frost_warning(
        self,
        frost_score: int,
        temperature_c: float,
        dewpoint_c: float,
        frost_margin_c: float,
        hour: int,
        confidence: float = 0.0,
    ) -> bool:
        """Show frost warning notification when heavy frost is detected @zara"""
        if not self._should_notify(CONF_NOTIFY_FROST):
            return False

        try:
            confidence_pct = int(confidence * 100)

            message = f"""**Starker Frost auf Solarpanelen erkannt!** ❄️

**Zeit:** {hour:02d}:00 Uhr
**Frost-Score:** {frost_score}/10
**Konfidenz:** {confidence_pct}%

**Wetterbedingungen:**
• Temperatur: {temperature_c:.1f}°C
• Taupunkt: {dewpoint_c:.1f}°C
• Frost-Margin: {frost_margin_c:.1f}°C

**Auswirkungen:**
• Die Solarproduktion ist wahrscheinlich reduziert
• Diese Stunde wird vom ML-Training ausgeschlossen
• Die Prognose-Genauigkeit kann beeinträchtigt sein

**Hinweis:** Frost löst sich normalerweise auf, sobald die Sonne die Panele erwärmt.

*"Even the coldest winter holds the promise of spring."* — Inspired by Star Trek"""

            await self._safe_create_notification(
                message=message,
                title="❄️ Frost auf Solarpanelen",
                notification_id=NOTIFICATION_ID_FROST,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing frost notification: {e}", exc_info=True)
            return False

    async def dismiss_frost_notification(self) -> bool:
        """Remove frost notification @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_FROST)

    async def show_weather_alert(
        self,
        alert_type: str,
        reason: str,
        hour: int,
        date_str: str,
        weather_actual: dict = None,
        weather_forecast: dict = None,
    ) -> bool:
        """Show weather alert notification when unexpected weather is detected @zara"""
        if not self._should_notify(CONF_NOTIFY_WEATHER_ALERT):
            return False

        try:
            # Map alert types to German descriptions
            alert_descriptions = {
                "unexpected_rain": "Unerwarteter Regen",
                "unexpected_snow": "Unerwarteter Schnee",
                "unexpected_clouds": "Unerwartete Bewölkung",
                "sudden_storm": "Plötzliches Unwetter",
                "unexpected_fog": "Unerwarteter Nebel",
            }
            alert_title = alert_descriptions.get(alert_type, alert_type)

            # Build weather details
            weather_details = ""
            if weather_actual:
                actual_rain = weather_actual.get("precipitation_mm", 0)
                actual_clouds = weather_actual.get("clouds", 0)
                actual_temp = weather_actual.get("temperature", "N/A")
                weather_details += f"""
**Aktuelle Wetterdaten:**
• Niederschlag: {actual_rain:.1f} mm
• Bewölkung: {actual_clouds}%
• Temperatur: {actual_temp}°C"""

            if weather_forecast:
                forecast_rain = weather_forecast.get("precipitation_probability", 0)
                forecast_clouds = weather_forecast.get("clouds", 0)
                weather_details += f"""

**Prognose war:**
• Niederschlagswahrscheinlichkeit: {forecast_rain}%
• Bewölkung: {forecast_clouds}%"""

            message = f"""**Unerwartetes Wetterereignis erkannt!** ⚠️

**Zeit:** {date_str} {hour:02d}:00 Uhr
**Ereignis:** {alert_title}
**Grund:** {reason}
{weather_details}

**Auswirkungen:**
• Die Solarproduktion weicht von der Prognose ab
• Diese Stunde wird vom ML-Training ausgeschlossen
• Die Prognose-Genauigkeit wird nicht beeinträchtigt

**Hinweis:** Das System lernt aus dieser Abweichung für zukünftige Wettervorhersagen.

*"Der Weltraum mag kalt sein, aber unsere Algorithmen lernen warm."* — Inspired by Star Trek 🖖"""

            await self._safe_create_notification(
                message=message,
                title=f"⚠️ Wetteralarm: {alert_title}",
                notification_id=NOTIFICATION_ID_WEATHER_ALERT,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing weather alert notification: {e}", exc_info=True)
            return False

    async def dismiss_weather_alert_notification(self) -> bool:
        """Remove weather alert notification @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_WEATHER_ALERT)

async def create_notification_service(
    hass: HomeAssistant, entry: ConfigEntry
) -> Optional[NotificationService]:
    """Factory function to create and initialize NotificationService"""
    try:
        service = NotificationService(hass, entry)

        if await service.initialize():
            _LOGGER.info("[OK] NotificationService created successfully")
            return service
        else:
            _LOGGER.warning("[!] NotificationService created but not initialized")
            return service

    except Exception as e:
        _LOGGER.error(f"[X] Failed to create NotificationService: {e}", exc_info=True)
        return None
