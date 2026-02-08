# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# *****************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# Refactored: JSON replaced with DatabaseManager @zara
# *****************************************************************************

"""
Notification service for Solar Forecast ML.
Handles persistent notifications in Home Assistant.
"""

import asyncio
import logging
from typing import List, Optional

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from ..const import (
    CONF_NOTIFY_FORECAST,
    CONF_NOTIFY_FOG,
    CONF_NOTIFY_FROST,
    CONF_NOTIFY_LEARNING,
    CONF_NOTIFY_STARTUP,
    CONF_NOTIFY_SUCCESSFUL_LEARNING,
    CONF_NOTIFY_WEATHER_ALERT,
    CONF_NOTIFY_SNOW_COVERED,
)

_LOGGER = logging.getLogger(__name__)

# Notification IDs
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
NOTIFICATION_ID_FOG = "solar_forecast_ml_fog"
NOTIFICATION_ID_WEATHER_ALERT = "solar_forecast_ml_weather_alert"
NOTIFICATION_ID_SNOW_COVERED = "solar_forecast_ml_snow_covered"
NOTIFICATION_ID_ADAPTIVE_CORRECTION = "solar_forecast_ml_adaptive_correction"


class NotificationService:
    """Service for Persistent Notifications in Home Assistant. @zara"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize Notification Service. @zara"""
        self.hass = hass
        self.entry = entry
        self._initialized = False
        self._notification_lock = asyncio.Lock()
        _LOGGER.debug("NotificationService instance created")

    async def initialize(self) -> bool:
        """Initialize the Notification Service. @zara"""
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
        """Centralized check if notification should be displayed. @zara"""
        if not self._initialized:
            return False

        enabled = self.entry.options.get(notification_type, True)

        if not enabled:
            _LOGGER.debug(f"Notification '{notification_type}' disabled by option")

        return enabled

    async def _safe_create_notification(
        self, message: str, title: str, notification_id: str
    ) -> bool:
        """Create notification with error handling. @zara"""
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
        """Remove notification with error handling. @zara"""
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

    async def create_notification(
        self, title: str, message: str, notification_id: str
    ) -> bool:
        """Public method to create a custom notification. @zara"""
        return await self._safe_create_notification(message, title, notification_id)

    async def dismiss_notification(self, notification_id: str) -> bool:
        """Public method to dismiss a notification. @zara"""
        return await self._safe_dismiss_notification(notification_id)

    async def show_startup_success(
        self,
        ml_mode: bool = True,
        installed_packages: Optional[List[str]] = None,
        missing_packages: Optional[List[str]] = None,
        use_attention: bool = False,
    ) -> bool:
        """Show startup notification with integration status. @zara"""
        if not self._should_notify(CONF_NOTIFY_STARTUP):
            return False

        try:
            installed_list = ""
            if installed_packages:
                installed_items = "\n".join([f"* {pkg}" for pkg in installed_packages])
                installed_list = f"\n\n**Installed Dependencies:**\n{installed_items}"

            missing_list = ""
            if missing_packages:
                missing_items = "\n".join([f"x {pkg}" for pkg in missing_packages])
                missing_list = f"\n\n**Missing Packages:**\n{missing_items}"

            if ml_mode:
                engine = "LSTM + Ridge Regression + Physics"
                if use_attention:
                    engine = "AI (Attention) + Ridge Regression + Physics"

                message = f"""Three proprietary AI models, a local Machine Learning engine, and a full solar physics engine work in perfect synergy to deliver **3-day hourly forecasts** with up to **97% accuracy** after calibration.

🔒 **100% local AI (3 + 1)** — no cloud, no subscriptions, no data leaves your network. No external AI (ChatGPT, Grok, Gemini) needed. Everything runs on your hardware.

| | |
|---|---|
| 🧠 **Mode** | Hybrid AI (Physics + Machine Learning) |
| ⚡ **Engine** | {engine} |
| 📊 **Features** | Multi-Panel · Weather · Self-Learning · Briefings |
| ✅ **Status** | All systems operational |
{installed_list}
> *"Logic is the beginning of wisdom, not the end."* — Spock

🖖 by **Zara-Toorox** — Live long and prosper!"""
            else:
                message = f"""A full solar physics engine delivers reliable forecasts — even without ML dependencies. Install the missing packages to unlock three proprietary AI models and up to **97% accuracy**.

🔒 **100% local** — no cloud, no subscriptions, no data leaves your network.

| | |
|---|---|
| 📐 **Mode** | Rule-Based (Limited Features) |
| 📊 **Features** | Solar Forecasting · Production Statistics |
| ✅ **Status** | Operational |
{missing_list}{installed_list}
⚠️ Install missing Python packages to enable ML features.

> *"Logic is the beginning of wisdom, not the end."* — Spock

🖖 by **Zara-Toorox**"""

            await self._safe_create_notification(
                message=message,
                title="☀️ Solar Forecast ML — Sarpeidion AI & DB-Version",
                notification_id=NOTIFICATION_ID_STARTUP,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing startup notification: {e}", exc_info=True)
            return False

    async def show_forecast_update(
        self, forecast_energy: float, confidence: Optional[float] = None
    ) -> bool:
        """Show forecast update notification. @zara"""
        if not self._should_notify(CONF_NOTIFY_FORECAST):
            return False

        try:
            confidence_text = ""
            if confidence is not None:
                confidence_text = f"\n**Confidence:** {confidence:.1f}%"

            message = f"Solar Forecast Updated"

            await self._safe_create_notification(
                message=message,
                title="Forecast Updated",
                notification_id=NOTIFICATION_ID_FORECAST,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing forecast notification: {e}", exc_info=True)
            return False

    async def show_training_start(self, sample_count: int) -> bool:
        """Show notification when AI training starts. @zara"""
        if not self._should_notify(CONF_NOTIFY_LEARNING):
            return False

        try:
            message = f"AI Training Started with {sample_count} samples"

            await self._safe_create_notification(
                message=message,
                title="Training Started",
                notification_id=NOTIFICATION_ID_LEARNING,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing training start notification: {e}", exc_info=True)
            return False

    async def show_training_complete(
        self, success: bool, accuracy: Optional[float] = None, sample_count: Optional[int] = None
    ) -> bool:
        """Show notification when AI training completes. @zara"""
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

                message = f"AI Training Complete{accuracy_text}{sample_text}"
            else:
                message = "AI Training Failed"

            await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)

            await self._safe_create_notification(
                message=message,
                title="Training Complete",
                notification_id=NOTIFICATION_ID_LEARNING,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing training complete notification: {e}", exc_info=True)
            return False

    async def dismiss_startup_notification(self) -> bool:
        """Remove startup notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_STARTUP)

    async def dismiss_forecast_notification(self) -> bool:
        """Remove forecast notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_FORECAST)

    async def dismiss_training_notification(self) -> bool:
        """Remove training notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)

    async def show_model_retraining_required(
        self,
        reason: str = "unknown",
        old_features: Optional[int] = None,
        new_features: Optional[int] = None,
    ) -> bool:
        """Show notification when AI model needs retraining. @zara"""
        try:
            if reason == "feature_mismatch":
                reason_text = f"""**Reason:** Sensor change detected

**Details:**
- Old Features: {old_features}
- New Features: {new_features}

The AI model will be automatically retrained to account for the changed sensor configuration."""
            else:
                reason_text = "The AI model needs to be retrained."

            message = f"""**Solar Forecast ML - Model Retraining Required**

{reason_text}

**Next Steps:**
- Training will be performed automatically
- If needed, manually start: Service `solar_forecast_ml.force_retrain`

**Status:** Automatic training running...

*"Adaptation is the key to survival."* - Inspired by Star Trek

**Personal Note from Zara:**
No worries! The integration adapts automatically."""

            await self._safe_create_notification(
                message=message,
                title="AI Model Retraining",
                notification_id=NOTIFICATION_ID_RETRAINING,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing retraining notification: {e}", exc_info=True)
            return False

    async def dismiss_retraining_notification(self) -> bool:
        """Remove retraining notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_RETRAINING)

    async def show_frost_warning(
        self,
        frost_score: int,
        temperature_c: float,
        dewpoint_c: Optional[float],
        frost_margin_c: float,
        hour: int,
        confidence: float = 0.0,
    ) -> bool:
        """Show frost warning notification when heavy frost is detected. @zara"""
        if not self._should_notify(CONF_NOTIFY_FROST):
            return False

        try:
            confidence_pct = int(confidence * 100)
            dewpoint_str = f"{dewpoint_c:.1f}°C" if dewpoint_c is not None else "N/A"
            frost_margin_str = f"{frost_margin_c:.1f}" if frost_margin_c is not None else "N/A"

            message = f"""**Heavy Frost on Solar Panels Detected!**

**Time:** {hour:02d}:00
**Frost Score:** {frost_score}/10
**Confidence:** {confidence_pct}%

**Weather Conditions:**
- Temperature: {temperature_c:.1f}°C
- Dew Point: {dewpoint_str}
- Frost Margin: {frost_margin_str}°C

**Effects:**
- Solar production is likely reduced
- This hour will be excluded from AI training
- Forecast accuracy may be affected

**Note:** Frost typically dissipates once the sun warms the panels.

*"Even the coldest winter holds the promise of spring."* - Inspired by Star Trek"""

            await self._safe_create_notification(
                message=message,
                title="Frost on Solar Panels",
                notification_id=NOTIFICATION_ID_FROST,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing frost notification: {e}", exc_info=True)
            return False

    async def dismiss_frost_notification(self) -> bool:
        """Remove frost notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_FROST)

    async def show_fog_warning(
        self,
        visibility_m: float,
        temperature_c: float,
        humidity: float,
        hour: int,
        fog_type: str = "dense",
    ) -> bool:
        """Show fog warning notification when dense fog is detected. @zara"""
        if not self._should_notify(CONF_NOTIFY_FOG):
            return False

        try:
            visibility_km = visibility_m / 1000.0
            fog_type_text = "Dense Fog" if fog_type == "dense" else "Light Fog"

            message = f"""**{fog_type_text} Detected!**

**Time:** {hour:02d}:00
**Visibility:** {visibility_km:.1f} km
**Humidity:** {humidity:.0f}%
**Temperature:** {temperature_c:.1f}C

**Effects on Solar Production:**
- Fog blocks less light than real clouds
- Diffuse radiation passes through fog
- The forecast is automatically adjusted

**Note:** Fog often dissipates once the sun gets stronger and temperature rises.

*"Through the fog, the sun still shines."* - Inspired by Star Trek"""

            await self._safe_create_notification(
                message=message,
                title="Dense Fog Detected",
                notification_id=NOTIFICATION_ID_FOG,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing fog notification: {e}", exc_info=True)
            return False

    async def dismiss_fog_notification(self) -> bool:
        """Remove fog notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_FOG)

    async def show_weather_alert(
        self,
        alert_type: str,
        reason: str,
        hour: int,
        date_str: str,
        weather_actual: dict = None,
        weather_forecast: dict = None,
    ) -> bool:
        """Show weather alert notification when unexpected weather is detected. @zara"""
        if not self._should_notify(CONF_NOTIFY_WEATHER_ALERT):
            return False

        try:
            alert_descriptions = {
                "unexpected_rain": "Unexpected Rain",
                "unexpected_snow": "Unexpected Snow",
                "unexpected_clouds": "Unexpected Clouds",
                "sudden_storm": "Sudden Storm",
                "unexpected_fog": "Unexpected Fog",
                "snow_covered_panels": "Snow Covered Panels",
            }
            alert_title = alert_descriptions.get(alert_type, alert_type)

            weather_details = ""
            if weather_actual:
                actual_rain = weather_actual.get("precipitation_mm", 0)
                actual_clouds = weather_actual.get("clouds", 0)
                actual_temp = weather_actual.get("temperature", "N/A")
                weather_details += f"""
**Current Weather Data:**
- Precipitation: {actual_rain:.1f} mm
- Cloud Cover: {actual_clouds}%
- Temperature: {actual_temp}C"""

            if weather_forecast:
                forecast_rain = weather_forecast.get("precipitation_probability", 0)
                forecast_clouds = weather_forecast.get("clouds", 0)
                weather_details += f"""

**Forecast Was:**
- Precipitation Probability: {forecast_rain}%
- Cloud Cover: {forecast_clouds}%"""

            message = f"""**Unexpected Weather Event Detected!**

**Time:** {date_str} {hour:02d}:00
**Event:** {alert_title}
**Reason:** {reason}
{weather_details}

**Effects:**
- Solar production deviates from forecast
- This hour will be excluded from AI training
- Forecast accuracy is not affected

**Note:** The system learns from this deviation for future weather forecasts.

*"Space may be cold, but our algorithms learn warm."* - Inspired by Star Trek"""

            await self._safe_create_notification(
                message=message,
                title=f"Weather Alert: {alert_title}",
                notification_id=NOTIFICATION_ID_WEATHER_ALERT,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing weather alert notification: {e}", exc_info=True)
            return False

    async def dismiss_weather_alert_notification(self) -> bool:
        """Remove weather alert notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_WEATHER_ALERT)

    async def show_snow_covered_warning(
        self,
        temperature_c: float,
        precipitation_mm: float,
        hour: int,
        message: str = None,
    ) -> bool:
        """Show warning when snow coverage on panels is possible. @zara V16.1"""
        if not self._should_notify(CONF_NOTIFY_SNOW_COVERED):
            return False

        try:
            estimated_depth = precipitation_mm * 8  # Conservative estimate

            if message:
                # Custom message provided (e.g., overnight snow) @zara V16.1
                notification_message = f"""❄️ **Schnee auf Solarmodulen erkannt**

**Zeit:** {hour:02d}:00 Uhr
**Temperatur:** {temperature_c:.1f}°C

{message}

**Hinweis:** Diese Stunden werden vom ML-Training ausgeschlossen.

*"Auch im kältesten Winter geht die Sonne auf."*"""
            else:
                notification_message = f"""❄️ **Schnee auf Solarmodulen möglich**

**Zeit:** {hour:02d}:00 Uhr
**Temperatur:** {temperature_c:.1f}°C
**Niederschlag:** {precipitation_mm:.1f} mm
**Geschätzte Schneehöhe:** ~{estimated_depth:.0f} mm

**Mögliche Auswirkungen:**
- Solarproduktion kann reduziert sein
- Diese Stunde wird vorsorglich vom ML-Training ausgeschlossen

**Hinweis:** Diese Warnung basiert auf Wetterdaten und ist eine Schätzung.

*"Auch im kältesten Winter geht die Sonne auf."*"""

            await self._safe_create_notification(
                message=notification_message,
                title="❄️ Schnee erkannt",
                notification_id=NOTIFICATION_ID_SNOW_COVERED,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing snow covered notification: {e}", exc_info=True)
            return False

    async def show_snow_melting_info(
        self,
        temperature_c: float,
        hour: int,
    ) -> bool:
        """Show info when snow may be melting from panels. @zara V16.1"""
        if not self._should_notify(CONF_NOTIFY_SNOW_COVERED):
            return False

        try:
            message = f"""☀️ **Schnee schmilzt vermutlich**

**Zeit:** {hour:02d}:00 Uhr
**Temperatur:** {temperature_c:.1f}°C

**Status:**
- Temperatur ist gestiegen
- Schnee beginnt zu schmelzen
- Solarproduktion normalisiert sich

**Hinweis:** Es kann noch einige Stunden dauern, bis die Module schneefrei sind.

*"Nach jedem Sturm kommt die Ruhe."*"""

            await self._safe_create_notification(
                message=message,
                title="☀️ Schnee schmilzt",
                notification_id=NOTIFICATION_ID_SNOW_COVERED,
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing snow melting notification: {e}", exc_info=True)
            return False

    async def dismiss_snow_covered_notification(self) -> bool:
        """Remove snow covered notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_SNOW_COVERED)

    async def show_adaptive_correction(
        self,
        original_kwh: float,
        corrected_kwh: float,
        reason: str,
        hours_corrected: int,
        am_actual: float = 0.0,
        am_predicted: float = 0.0,
    ) -> bool:
        """Show notification when adaptive forecast correction was applied. @zara"""
        try:
            # Calculate change percentage
            if original_kwh > 0.1:
                change_percent = ((corrected_kwh - original_kwh) / original_kwh) * 100
                change_direction = "+" if change_percent > 0 else ""
                change_text = f"{change_direction}{change_percent:.0f}%"
            else:
                change_text = "N/A"

            # Calculate morning deviation
            if am_predicted > 0.1:
                am_deviation = ((am_actual - am_predicted) / am_predicted) * 100
                am_deviation_text = f"{am_deviation:+.0f}%"
            else:
                am_deviation_text = "N/A"

            message = f"""**Adaptive Forecast Correction Applied**

**Reason:** {reason}

**Morning Analysis:**
- Actual Production: {am_actual:.2f} kWh
- Forecast Was: {am_predicted:.2f} kWh
- Deviation: {am_deviation_text}

**Correction:**
- Original Daily Forecast: {original_kwh:.2f} kWh
- Corrected Daily Forecast: {corrected_kwh:.2f} kWh ({change_text})
- Recalculated Hours: {hours_corrected}

**Note:**
The afternoon forecast was recalculated based on more recent weather data.
Morning values remain unchanged.

*"Adaptation is the key to survival."* - Inspired by Star Trek"""

            await self._safe_create_notification(
                message=message,
                title="Forecast Automatically Adjusted",
                notification_id=NOTIFICATION_ID_ADAPTIVE_CORRECTION,
            )

            return True

        except Exception as e:
            _LOGGER.error(
                f"[X] Error showing adaptive correction notification: {e}", exc_info=True
            )
            return False

    async def dismiss_adaptive_correction_notification(self) -> bool:
        """Remove adaptive correction notification. @zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_ADAPTIVE_CORRECTION)


async def create_notification_service(
    hass: HomeAssistant, entry: ConfigEntry
) -> Optional[NotificationService]:
    """Factory function to create and initialize NotificationService. @zara"""
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
