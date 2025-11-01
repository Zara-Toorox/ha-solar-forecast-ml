"""
Notification Service for Solar Forecast ML Integration.
✅ PRODUCTION READY: Async Factory Pattern with Option Validation
✅ FIX: Centralized Notification Option Checking
✅ NEW: show_forecast_update, show_training_start, show_training_complete
Version 5.0.0

Copyright (C) 2025 Zara-Toorox

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
    ✅ ASYNC: All methods non-blocking
    ✅ SAFE: Error handling for all notification operations
    ✅ CENTRAL: Centralized option checking before notifications
    """
    
    def __init__(self, hass: HomeAssistant, entry: ConfigEntry):
        """
        Initialize Notification Service.
        
        ⚠️ DO NOT CALL DIRECTLY: Use create_notification_service()
        
        Args:
            hass: HomeAssistant instance
            entry: ConfigEntry for option access
        """
        self.hass = hass
        self.entry = entry
        self._initialized = False
        self._notification_lock = asyncio.Lock()
        _LOGGER.debug("🔧 NotificationService instance created")
    
    async def initialize(self) -> bool:
        """
        Initialize the Notification Service.
        
        Returns:
            True if successfully initialized
        """
        try:
            async with self._notification_lock:
                if self._initialized:
                    _LOGGER.debug("✓ NotificationService already initialized")
                    return True
                
                # Check if persistent_notification component is loaded
                if 'persistent_notification' not in self.hass.config.components:
                    _LOGGER.warning(
                        "⚠️ persistent_notification not available - "
                        "Notifications will not be displayed"
                    )
                    self._initialized = True
                    return False
                
                self._initialized = True
                _LOGGER.info("✅ NotificationService successfully initialized")
                return True
                
        except Exception as e:
            _LOGGER.error(
                f"❌ Error during NotificationService initialization: {e}",
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
    
    # ========================================================================
    # 🔧 FIX: New import mechanism without None problems
    # ========================================================================
    async def _safe_create_notification(
        self,
        message: str,
        title: str,
        notification_id: str
    ) -> bool:
        """
        Create notification with error handling.
        
        ✅ FIX: Uses hass.services.async_call instead of direct import
        """
        if not self._initialized:
            _LOGGER.warning(
                f"⚠️ NotificationService not initialized - "
                f"Notification '{notification_id}' will not be displayed"
            )
            return False
        
        try:
            # ✅ FIX: Use hass.services.async_call instead of problematic import
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
            _LOGGER.debug(f"🔧 Notification '{notification_id}' created")
            return True
            
        except Exception as e:
            _LOGGER.error(
                f"❌ Error creating notification '{notification_id}': {e}",
                exc_info=True
            )
            return False
    
    async def _safe_dismiss_notification(self, notification_id: str) -> bool:
        """
        Remove notification with error handling.
        
        ✅ FIX: Uses hass.services.async_call
        """
        if not self._initialized:
            return False
        
        try:
            # ✅ FIX: Use hass.services.async_call
            await self.hass.services.async_call(
                domain="persistent_notification",
                service="dismiss",
                service_data={
                    "notification_id": notification_id,
                },
                blocking=True,
            )
            _LOGGER.debug(f"🔧 Notification '{notification_id}' dismissed")
            return True
            
        except Exception as e:
            _LOGGER.warning(
                f"⚠️ Error dismissing notification '{notification_id}': {e}"
            )
            return False
    # ========================================================================
    # END FIX
    # ========================================================================
    
    async def show_startup_success(
        self, 
        ml_mode: bool = True,
        installed_packages: Optional[List[str]] = None,
        missing_packages: Optional[List[str]] = None
    ) -> bool:
        """
        Show startup notification with integration status.
        
        ✅ Option Check: Respects CONF_NOTIFY_STARTUP
        
        Args:
            ml_mode: True if ML features active, False on fallback
            installed_packages: List of installed packages (informative)
            missing_packages: Optional - List of missing packages on fallback
            
        Returns:
            True if successful
        """
        # ✅ Centralized option check
        if not self._should_notify(CONF_NOTIFY_STARTUP):
            return False
            
        try:
            # Create list of installed dependencies (informative)
            installed_list = ""
            if installed_packages:
                installed_items = "\n".join([f"- ✅ {pkg}" for pkg in installed_packages])
                installed_list = f"\n**Installed Dependencies:**\n{installed_items}\n"
            
            if ml_mode:
                # ✅ ML Mode active - all dependencies present
                message = f"""
**🎯 Solar Forecast ML 6 RISA Started!**

**Status: Full ML Mode ✅**

Integration running with all features:
{installed_list}
✅ **Machine Learning active**
- Ridge Regression Model
- 28 Features for predictions
- Automatic model training
- Pattern recognition

✅ **Forecast System**
- Hourly forecasts
- Weather integration
- Production tracking

✅ **Smart Features**
- Seasonal adjustments
- Weather-based optimization
- Continuous learning

**Next Steps:**
The ML model will automatically train with real data and continuously improve.
You can manually trigger training via the button entity.

Integration is ready!
"""
            else:
                # ⚠️ Fallback Mode - Dependencies missing
                missing_list = ""
                if missing_packages:
                    missing_items = "\n".join([f"- ❌ {pkg}" for pkg in missing_packages])
                    missing_list = f"\n**Missing Packages:**\n{missing_items}\n"
                
                message = f"""
**⚠️ Solar Forecast ML Started in Fallback Mode**

**Status: Rule-Based Mode (ML disabled)**
{installed_list}{missing_list}
**Why Fallback Mode?**
Required ML dependencies (numpy, aiofiles) could not be installed automatically.

**Available Features:**
✅ Basic forecasts (rule-based)
✅ Weather integration
✅ Production tracking

**Missing Features:**
❌ Machine Learning predictions
❌ Pattern recognition
❌ Automatic model improvement

**Install ML Dependencies:**
You can install dependencies manually:

**Via SSH (Docker):**
```
docker exec homeassistant pip install --break-system-packages numpy aiofiles
```

**Or via Add-on Terminal:**
```
pip install numpy aiofiles
```

After installation, restart Home Assistant for full ML mode.

**Need Help?**
Check logs for details or contact support.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="🎯 Solar Forecast ML - Started",
                notification_id=NOTIFICATION_ID_STARTUP
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_startup_success: {e}", exc_info=True)
            return False
    
    async def show_dependency_missing(self, missing_packages: List[str]) -> bool:
        """
        Show notification about missing dependencies with installation instructions.
        
        Args:
            missing_packages: List of missing package names
            
        Returns:
            True if successful
        """
        try:
            package_list = "\n".join([f"- {pkg}" for pkg in missing_packages])
            
            message = f"""
**⚠️ ML Dependencies Missing**

Solar Forecast ML requires the following packages:

{package_list}

**Current Status:**
Integration running in **Fallback Mode** (rule-based predictions only).

**To Enable ML Features:**

**Option 1: Automatic Installation (Recommended)**
Use the "Install ML Dependencies" button that will appear in the integration.

**Option 2: Manual Installation**
Connect via SSH to Home Assistant and run:

**For Docker installations:**
```
docker exec homeassistant pip install --break-system-packages numpy aiofiles
```

**For other installations:**
```
pip install numpy aiofiles
```

**After installation:**
Restart Home Assistant to activate ML features.

**Need Help?**
Check the logs for detailed error messages or contact support.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="⚠️ Solar Forecast ML - Dependencies Missing",
                notification_id=NOTIFICATION_ID_DEPENDENCIES
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_dependency_missing: {e}", exc_info=True)
            return False
    
    async def show_forecast_update(
        self,
        next_hour_value: float,
        daily_total: float,
        strategy: str = "ML"
    ) -> bool:
        """
        Show notification when forecast is updated.
        
        ✅ Option Check: Respects CONF_NOTIFY_FORECAST
        
        Args:
            next_hour_value: Forecasted production for next hour in kWh
            daily_total: Total forecasted production for today in kWh
            strategy: Strategy used (ML/Weather/Rule)
            
        Returns:
            True if successful
        """
        # ✅ Centralized option check
        if not self._should_notify(CONF_NOTIFY_FORECAST):
            return False
            
        try:
            strategy_icon = "🎯" if strategy == "ML" else "☀️"
            
            message = f"""
**{strategy_icon} Forecast Updated**

**Next Hour:** {next_hour_value:.2f} kWh
**Today Total:** {daily_total:.2f} kWh

**Strategy:** {strategy}

Last update: just now
"""
            
            return await self._safe_create_notification(
                message=message,
                title=f"{strategy_icon} Solar Forecast - Updated",
                notification_id=NOTIFICATION_ID_FORECAST
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_forecast_update: {e}", exc_info=True)
            return False
    
    async def show_training_start(self, sample_count: int) -> bool:
        """
        Show notification when ML training starts.
        
        ✅ Option Check: Respects CONF_NOTIFY_LEARNING
        
        Args:
            sample_count: Number of training samples
            
        Returns:
            True if successful
        """
        # ✅ Centralized option check
        if not self._should_notify(CONF_NOTIFY_LEARNING):
            return False
            
        try:
            message = f"""
**🎯 ML Training Started**

**Training Samples:** {sample_count}

The ML model is being trained with your production data.
This may take a few moments...

**What happens:**
- Feature extraction
- Model training
- Validation
- Performance check

Training in progress...
"""
            
            return await self._safe_create_notification(
                message=message,
                title="🎯 Solar Forecast ML - Training",
                notification_id=NOTIFICATION_ID_LEARNING
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_training_start: {e}", exc_info=True)
            return False
    
    async def show_training_complete(
        self,
        samples_used: int,
        r2_score: Optional[float] = None,
        mae: Optional[float] = None
    ) -> bool:
        """
        Show notification when ML training completes successfully.
        
        ✅ Option Check: Respects CONF_NOTIFY_SUCCESSFUL_LEARNING
        
        Args:
            samples_used: Number of samples used for training
            r2_score: Optional R² score (model quality)
            mae: Optional Mean Absolute Error
            
        Returns:
            True if successful
        """
        # ✅ Centralized option check
        if not self._should_notify(CONF_NOTIFY_SUCCESSFUL_LEARNING):
            return False
            
        try:
            # Dismiss training notification
            await self._safe_dismiss_notification(NOTIFICATION_ID_LEARNING)
            
            # Build metrics section
            metrics = ""
            if r2_score is not None:
                quality = "Excellent" if r2_score > 0.8 else "Good" if r2_score > 0.6 else "Fair"
                metrics += f"\n**Model Quality:** {quality} (R² = {r2_score:.3f})"
            
            if mae is not None:
                metrics += f"\n**Accuracy:** ±{mae:.2f} kWh (MAE)"
            
            message = f"""
**✅ ML Training Complete!**

**Training Samples:** {samples_used}
{metrics}

**Status:** Model active and ready

**What now?**
The trained model is now used for forecasts.
It will continuously improve with more data.

**Next training:**
Automatically scheduled or manually via button.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="✅ Solar Forecast ML - Training Complete",
                notification_id=NOTIFICATION_ID_LEARNING
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_training_complete: {e}", exc_info=True)
            return False
    
    async def show_installation_progress(self, status: str, progress: int) -> bool:
        """
        Show installation progress as notification.
        
        Args:
            status: Status text
            progress: Progress 0-100
            
        Returns:
            True if successful
        """
        try:
            # Progress bar with Unicode characters
            bar_length = 20
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            message = f"""
**🔧 Installation Running...**

{bar} {progress}%

**Status:** {status}

Please wait until installation is complete.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="🔧 Solar Forecast ML - Installation",
                notification_id=NOTIFICATION_ID_INSTALLATION
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_installation_progress: {e}", exc_info=True)
            return False
    
    async def show_installation_success(self) -> bool:
        """
        Show success notification after successful installation.
        
        Returns:
            True if successful
        """
        try:
            # Dismiss progress notification
            await self._safe_dismiss_notification(NOTIFICATION_ID_INSTALLATION)
            
            message = """
**✅ Installation Successful!**

All ML dependencies successfully installed:
- ✅ numpy installed
- ✅ aiofiles installed

**⚠️ Important: Restart Required**

Please restart Home Assistant to activate ML features.

After restart:
- ML model will automatically train
- Enhanced forecasts available
- Pattern recognition active

Integration will run in **Full ML Mode**.
"""
            
            # Show new success notification
            return await self._safe_create_notification(
                message=message,
                title="✅ Solar Forecast ML - Installation Successful",
                notification_id=NOTIFICATION_ID_SUCCESS
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_installation_success: {e}", exc_info=True)
            return False
    
    async def show_installation_error(
        self, 
        error_message: str,
        failed_packages: Optional[List[str]] = None
    ) -> bool:
        """
        Show error notification on failed installation.
        
        Args:
            error_message: Error description
            failed_packages: Optional - List of failed packages
            
        Returns:
            True if successful
        """
        try:
            # Dismiss progress notification
            await self._safe_dismiss_notification(NOTIFICATION_ID_INSTALLATION)
            
            failed_list = ""
            if failed_packages:
                failed_list = "\n".join([f"- {pkg}" for pkg in failed_packages])
                failed_list = f"\n**Failed Packages:**\n{failed_list}\n"
            
            message = f"""
**❌ Installation Failed**

{error_message}
{failed_list}

**Manual Installation:**

Connect via SSH to Home Assistant and run:

```
docker exec homeassistant pip install --break-system-packages numpy aiofiles
```

Or if not using Docker:
```
pip install numpy aiofiles
```

Then restart Home Assistant.

**Need Help?**
Check logs for details or contact support.
"""
            
            return await self._safe_create_notification(
                message=message,
                title="❌ Solar Forecast ML - Installation Failed",
                notification_id=NOTIFICATION_ID_ERROR
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_installation_error: {e}", exc_info=True)
            return False
    
    async def show_ml_activated(self) -> bool:
        """
        Show notification when ML is successfully activated.
        
        Returns:
            True if successful
        """
        try:
            message = """
**🎯 ML Features Activated!**

Solar Forecast ML now running in **Full ML Mode**:

✅ Ridge Regression Model active
✅ 28 Features for forecasts
✅ Automatic learning active
✅ Pattern recognition active
✅ Seasonal adjustments

**What does this mean?**

Integration now uses Machine Learning for:
- More precise predictions
- Adaptation to your system
- Weather-based optimization
- Continuous improvement

Model automatically trains with real data and improves over time.

**Status:** ML is operational
"""
            
            return await self._safe_create_notification(
                message=message,
                title="🎯 Solar Forecast ML - ML Activated",
                notification_id=NOTIFICATION_ID_ML_ACTIVE
            )
            
        except Exception as e:
            _LOGGER.error(f"❌ Error in show_ml_activated: {e}", exc_info=True)
            return False
    
    async def dismiss_all(self) -> None:
        """
        Remove all Solar Forecast ML notifications.
        
        Useful for cleanup.
        """
        try:
            notification_ids = [
                NOTIFICATION_ID_DEPENDENCIES,
                NOTIFICATION_ID_INSTALLATION,
                NOTIFICATION_ID_SUCCESS,
                NOTIFICATION_ID_ERROR,
                NOTIFICATION_ID_ML_ACTIVE,
                NOTIFICATION_ID_STARTUP,
                NOTIFICATION_ID_FORECAST,
                NOTIFICATION_ID_LEARNING,
            ]
            
            for notification_id in notification_ids:
                await self._safe_dismiss_notification(notification_id)
                
            _LOGGER.debug("✅ All notifications dismissed")
            
        except Exception as e:
            _LOGGER.warning(f"⚠️ Error dismissing all notifications: {e}")


async def create_notification_service(
    hass: HomeAssistant,
    entry: ConfigEntry
) -> NotificationService:
    """
    Factory function for creating a NotificationService.
    
    ✅ ASYNC: Correct initialization
    ✅ SAFE: Error handling
    ✅ NEW: Passes entry for option checking
    
    Args:
        hass: HomeAssistant instance
        entry: ConfigEntry for option access
        
    Returns:
        Initialized NotificationService
        
    Raises:
        Exception: On critical initialization errors
    """
    try:
        service = NotificationService(hass, entry)
        await service.initialize()
        return service
        
    except Exception as e:
        _LOGGER.error(
            f"❌ Error creating NotificationService: {e}",
            exc_info=True
        )
        # Return service anyway, but not initialized
        # Integration can continue without notifications
        return NotificationService(hass, entry)
