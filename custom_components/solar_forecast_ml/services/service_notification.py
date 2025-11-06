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


class NotificationService:
    """
    Service for Persistent Notifications in Home Assistant.
    All methods are non-blocking with proper error handling.
    Centralized option checking before notifications.
    """

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """
        Initialize Notification Service.

        WARNING: Do not call directly - use create_notification_service()

        Args:
            hass: HomeAssistant instance
            entry: ConfigEntry for option access
        """
        self.hass = hass
        self.entry = entry
        self._initialized = False
        self._notification_lock = asyncio.Lock()
        _LOGGER.debug("NotificationService instance created")

    async def initialize(self) -> bool:
        """
        Initialize the Notification Service.

        Returns:
            True if successfully initialized
        """
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
        """
        Centralized check if notification should be displayed.

        Args:
            notification_type: CONF_NOTIFY_* constant

        Returns:
            True if notification is allowed
        """
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
        """
        Create notification with error handling.
        Uses hass.services.async_call instead of direct import.
        """
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
        """
        Remove notification with error handling.
        """
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
        """
        Show startup notification with integration status.

        Args:
            ml_mode: True if ML features active, False on fallback
            installed_packages: List of installed packages
            missing_packages: List of missing packages on fallback

        Returns:
            True if successful
        """
        if not self._should_notify(CONF_NOTIFY_STARTUP):
            return False

        try:
            installed_list = ""
            if installed_packages:
                installed_items = "\n".join([f"- [OK] {pkg}" for pkg in installed_packages])
                installed_list = f"\n**Installed Dependencies:**\n{installed_items}\n"

            if ml_mode:
                message = f"""
**Solar Forecast ML Started Successfully**

**Status: Full ML Mode [OK]**

Integration running with all features:
{installed_list}
[OK] **Machine Learning active**
- Ridge Regression Model
- 28 Features for predictions
- Automatic model training
- Pattern recognition

[OK] **Forecast System**
- Hourly forecasts
- Weather integration
- Production tracking

[OK] **Smart Features**
- Seasonal adjustments
- Weather-based optimization
- Continuous learning

**Next Steps:**
The ML model will automatically train with real data and continuously improve.
You can manually trigger training via the button entity.

Integration is ready!
"""
            else:
                missing_list = ""
                if missing_packages:
                    missing_items = "\n".join([f"- [X] {pkg}" for pkg in missing_packages])
                    missing_list = f"\n**Missing Packages:**\n{missing_items}\n"

                message = f"""
**[!] Solar Forecast ML Started in Fallback Mode**

**Status: Rule-Based Mode (ML disabled)**
{installed_list}{missing_list}
**Why Fallback Mode?**
Required ML dependencies (numpy, aiofiles) could not be installed automatically.

**Available Features:**
[OK] Basic forecasts (rule-based)
[OK] Weather integration
[OK] Production tracking

**Missing Features:**
[X] Machine Learning predictions
[X] Pattern recognition
[X] Automatic model improvement

**Install ML Dependencies:**
You can install dependencies manually:

**Via SSH (Docker):**
```bash
docker exec homeassistant pip install numpy aiofiles scikit-learn
```

**Via Terminal Add-on:**
```bash
pip install numpy aiofiles scikit-learn
```

After installation, restart Home Assistant to enable ML mode.
"""

            await self._safe_create_notification(
                message=message,
                title="Solar Forecast ML Started",
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
        """Show forecast update notification."""
        if not self._should_notify(CONF_NOTIFY_FORECAST):
            return False

        try:
            confidence_text = ""
            if confidence is not None:
                confidence_text = f"\n**Confidence:** {confidence:.1f}%"

            message = f"""
**Solar Forecast Updated**

**Today's Forecast:** {forecast_energy:.2f} kWh{confidence_text}

The forecast has been updated based on current weather conditions.
"""

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
        """Show notification when ML training starts."""
        if not self._should_notify(CONF_NOTIFY_LEARNING):
            return False

        try:
            message = f"""
**ML Training Started**

**Samples:** {sample_count}

The machine learning model is being trained with historical data.
This may take a few moments.
"""

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
        """Show notification when ML training completes."""
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

                message = f"""
**[OK] ML Training Complete**

The machine learning model has been successfully trained and is now active.{accuracy_text}{sample_text}

Future forecasts will use the trained model for improved accuracy.
"""
            else:
                message = """
**[!] ML Training Failed**

The training process encountered an error. The system will continue using rule-based forecasts.

Check the logs for more details.
"""

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
        """Remove startup notification."""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_STARTUP)

    async def dismiss_forecast_notification(self) -> bool:
        """Remove forecast notification."""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_FORECAST)

    async def dismiss_training_notification(self) -> bool:
        """Remove training notification."""
        return await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)


async def create_notification_service(
    hass: HomeAssistant,
    entry: ConfigEntry
) -> Optional[NotificationService]:
    """
    Factory function to create and initialize NotificationService.

    Args:
        hass: HomeAssistant instance
        entry: ConfigEntry for option access

    Returns:
        Initialized NotificationService or None on failure
    """
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
