"""Solar Forecast ML Integration."""
import logging
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Solar Forecast ML from a config entry."""
    _LOGGER.info("Setting up Solar Forecast ML integration")
    
    hass.data.setdefault(DOMAIN, {})
    
    # Lazy import coordinator to reduce startup blocking
    from .coordinator import SolarForecastMLCoordinator
    
    # Initialize coordinator
    coordinator = SolarForecastMLCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    
    # Store coordinator instance
    hass.data[DOMAIN][entry.entry_id] = coordinator
    
    # Forward setup to platforms
    await hass.config_entries.async_forward_entry_setups(entry, ["sensor", "button"])
    
    # Send startup notification after successful setup
    try:
        from .services.service_notification import create_notification_service
        notification_service = await create_notification_service(hass, entry)
        
        if notification_service:
            ml_mode = coordinator._ml_ready if hasattr(coordinator, '_ml_ready') else False
            await notification_service.show_startup_success(
                ml_mode=ml_mode,
                installed_packages=None,
                missing_packages=None
            )
            _LOGGER.debug("Startup notification sent successfully")
    except Exception as notif_err:
        _LOGGER.warning(f"Failed to send startup notification (non-critical): {notif_err}")
    
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, ["sensor", "button"])
    
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id)
    
    return unload_ok
