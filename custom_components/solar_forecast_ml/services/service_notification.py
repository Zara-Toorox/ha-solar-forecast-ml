"""
Notification Service for Solar Forecast ML Integration

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

import logging
import asyncio
from typing import Optional, List
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry

from ..const import (
    CONF_NOTIFY_STARTUP,
    CONF_NOTIFY_FORECAST,
    CONF_NOTIFY_LEARNING,
    CONF_NOTIFY_SUCCESSFUL_LEARNING,
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


class NotificationService:
    """Service for Persistent Notifications in Home Assistant by @Zara"""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """Initialize Notification Service by @Zara"""
        self.hass = hass
        self.entry = entry
        self._initialized = False
        self._notification_lock = asyncio.Lock()
        _LOGGER.debug("NotificationService instance created")

    async def initialize(self) -> bool:
        """Initialize the Notification Service by @Zara"""
        try:
            async with self._notification_lock:
                if self._initialized:
                    _LOGGER.debug("[OK] NotificationService already initialized")
                    return True

                # Check if persistent_notification component is loaded
                if 'persistent_notification' not in self.hass.config.components:
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
                f"[X] Error during NotificationService initialization: {e}",
                exc_info=True
            )
            return False

    def _should_notify(self, notification_type: str) -> bool:
        """Centralized check if notification should be displayed by @Zara"""
        if not self._initialized:
            return False

        # Get option from entry.options with fallback to default
        enabled = self.entry.options.get(notification_type, True)

        if not enabled:
            _LOGGER.debug(f"Notification '{notification_type}' disabled by option")

        return enabled

    async def _safe_create_notification(
        self,
        message: str,
        title: str,
        notification_id: str
    ) -> bool:
        """Create notification with error handling by @Zara"""
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
                f"[X] Error creating notification '{notification_id}': {e}",
                exc_info=True
            )
            return False

    async def _safe_dismiss_notification(self, notification_id: str) -> bool:
        """Remove notification with error handling by @Zara"""
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
            _LOGGER.warning(
                f"[!] Error dismissing notification '{notification_id}': {e}"
            )
            return False

    async def show_startup_success(
        self,
        ml_mode: bool = True,
        installed_packages: Optional[List[str]] = None,
        missing_packages: Optional[List[str]] = None
    ) -> bool:
        """Show startup notification with integration status by @Zara"""
        if not self._should_notify(CONF_NOTIFY_STARTUP):
            return False

        try:
            # Build dependency list
            installed_list = ""
            if installed_packages:
                installed_items = "\n".join([f"✓ {pkg}" for pkg in installed_packages])
                installed_list = f"\n\n**Installed Dependencies:**\n{installed_items}"

            missing_list = ""
            if missing_packages:
                missing_items = "\n".join([f"✗ {pkg}" for pkg in missing_packages])
                missing_list = f"\n\n**Missing Packages:**\n{missing_items}"

            # Create personalized startup message
            if ml_mode:
                message = f"""**Solar Forecast ML v8.2.6 "Sarpeidon" Started Successfully!** ☀️

**Mode:** Machine Learning (Full Feature Set)

**Version:** 8.2.6 "Sarpeidon" - Named after the doomed planet whose inhabitants used time portals to escape

**Author:** Zara-Toorox

**Active Features:**
• ML-based solar production forecasting
• Historical data analysis & learning
• Weather integration for accuracy
• Peak production time detection
• Autarky & self-sufficiency tracking
• Battery management & cost optimization{installed_list}

**System Status:** All systems operational ✓

*"The future is not set in stone, but with data and logic, we can illuminate the path ahead."* — Inspired by Star Trek

**Personal Note from Zara:**
Thank you for using Solar Forecast ML! May your panels generate efficiently and your batteries stay charged. Live long and prosper! 🖖"""
            else:
                message = f"""**Solar Forecast ML v8.2.6 "Sarpeidon" Started in Fallback Mode** ⚠️

**Mode:** Rule-Based Calculations (Limited Features)

**Version:** 8.2.6 "Sarpeidon"

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
                notification_id=NOTIFICATION_ID_STARTUP
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing startup notification: {e}", exc_info=True)
            return False

    async def show_forecast_update(
        self,
        forecast_energy: float,
        confidence: Optional[float] = None
    ) -> bool:
        """Show forecast update notification by @Zara"""
        if not self._should_notify(CONF_NOTIFY_FORECAST):
            return False

        try:
            confidence_text = ""
            if confidence is not None:
                confidence_text = f"\n**Confidence:** {confidence:.1f}%"

            message = f"""Solar Forecast Updated by @Zara"""

            await self._safe_create_notification(
                message=message,
                title="Forecast Updated",
                notification_id=NOTIFICATION_ID_FORECAST
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing forecast notification: {e}", exc_info=True)
            return False

    async def show_training_start(self, sample_count: int) -> bool:
        """Show notification when ML training starts by @Zara"""
        if not self._should_notify(CONF_NOTIFY_LEARNING):
            return False

        try:
            message = f"""ML Training Started by @Zara"""

            await self._safe_create_notification(
                message=message,
                title="Training Started",
                notification_id=NOTIFICATION_ID_LEARNING
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing training start notification: {e}", exc_info=True)
            return False

    async def show_training_complete(
        self,
        success: bool,
        accuracy: Optional[float] = None,
        sample_count: Optional[int] = None
    ) -> bool:
        """Show notification when ML training completes by @Zara"""
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

                message = f"""OK ML Training Complete by @Zara"""
            else:
                message = """ ML Training Failed by @Zara"""

            await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)

            await self._safe_create_notification(
                message=message,
                title="Training Complete",
                notification_id=NOTIFICATION_ID_LEARNING
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing training complete notification: {e}", exc_info=True)
            return False

    async def dismiss_startup_notification(self) -> bool:
        """Remove startup notification by @Zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_STARTUP)

    async def dismiss_forecast_notification(self) -> bool:
        """Remove forecast notification by @Zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_FORECAST)

    async def dismiss_training_notification(self) -> bool:
        """Remove training notification by @Zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)

    async def show_model_retraining_required(
        self,
        reason: str = "unknown",
        old_features: Optional[int] = None,
        new_features: Optional[int] = None
    ) -> bool:
        """Show notification when ML model needs retraining by @Zara"""
        try:
            # Build reason-specific message
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
                notification_id=NOTIFICATION_ID_RETRAINING
            )

            return True

        except Exception as e:
            _LOGGER.error(f"[X] Error showing retraining notification: {e}", exc_info=True)
            return False

    async def dismiss_retraining_notification(self) -> bool:
        """Remove retraining notification by @Zara"""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_RETRAINING)


async def create_notification_service(
    hass: HomeAssistant,
    entry: ConfigEntry
) -> Optional[NotificationService]:
    """Factory function to create and initialize NotificationService by @Zara"""
    try:
        service = NotificationService(hass, entry)

        if await service.initialize():
            _LOGGER.info("[OK] NotificationService created successfully")
            return service
        else:
            _LOGGER.warning("[!] NotificationService created but not initialized")
            return service

    except Exception as e:
        _LOGGER.error(
            f"[X] Failed to create NotificationService: {e}",
            exc_info=True
        )
        return None
