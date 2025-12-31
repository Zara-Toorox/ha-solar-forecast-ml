# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.config_entries import ConfigEntry, SOURCE_RECONFIGURE
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.helpers import selector

from .const import (
    CONF_BATTERY_POWER_SENSOR,
    CONF_CALIBRATION_PRICE,
    CONF_COUNTRY,
    CONF_GRID_FEE,
    CONF_MAX_PRICE,
    CONF_PROVIDER_MARKUP,
    CONF_TAXES_FEES,
    CONF_USE_CALIBRATION,
    CONF_VAT_RATE,
    COUNTRY_OPTIONS,
    DEFAULT_COUNTRY,
    DEFAULT_GRID_FEE,
    DEFAULT_MAX_PRICE,
    DEFAULT_PROVIDER_MARKUP,
    DEFAULT_TAXES_FEES,
    DOMAIN,
    NAME,
    VAT_OPTIONS,
    VAT_RATE_AT,
    VAT_RATE_DE,
)

# Note: ElectricityPriceService is imported lazily in async_step_pricing
# to avoid blocking the event loop during module import

_LOGGER = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _get_default(data: dict | None, key: str, default: Any = vol.UNDEFINED) -> Any:
    """Safely get default value for schema @zara"""
    if data is None:
        return default
    value = data.get(key)
    return value if value is not None and value != "" else default


def _get_default_vat_for_country(country: str) -> int:
    """Get default VAT rate for country @zara"""
    return VAT_RATE_AT if country == "AT" else VAT_RATE_DE


def _get_country_schema(default_country: str = DEFAULT_COUNTRY) -> vol.Schema:
    """Returns the country selection schema @zara"""
    return vol.Schema({
        vol.Required(
            CONF_COUNTRY,
            default=default_country,
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value=k, label=v)
                    for k, v in COUNTRY_OPTIONS.items()
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
            ),
        ),
    })


def _get_pricing_schema(
    defaults: dict | None = None,
    default_vat: int = VAT_RATE_DE,
) -> vol.Schema:
    """Returns the pricing configuration schema @zara"""
    if defaults is None:
        defaults = {}

    return vol.Schema({
        vol.Required(
            CONF_VAT_RATE,
            default=defaults.get(CONF_VAT_RATE, default_vat),
        ): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value=str(opt["value"]), label=opt["label"])
                    for opt in VAT_OPTIONS
                ],
                mode=selector.SelectSelectorMode.DROPDOWN,
            ),
        ),
        vol.Optional(
            CONF_USE_CALIBRATION,
            default=False,
        ): selector.BooleanSelector(),
        vol.Optional(
            CONF_CALIBRATION_PRICE,
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                max=100,
                step=0.01,
                unit_of_measurement="ct/kWh",
                mode=selector.NumberSelectorMode.BOX,
            ),
        ),
        vol.Required(
            CONF_GRID_FEE,
            default=defaults.get(CONF_GRID_FEE, DEFAULT_GRID_FEE),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                max=50,
                step=0.01,
                unit_of_measurement="ct/kWh",
                mode=selector.NumberSelectorMode.BOX,
            ),
        ),
        vol.Required(
            CONF_TAXES_FEES,
            default=defaults.get(CONF_TAXES_FEES, DEFAULT_TAXES_FEES),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                max=50,
                step=0.01,
                unit_of_measurement="ct/kWh",
                mode=selector.NumberSelectorMode.BOX,
            ),
        ),
        vol.Required(
            CONF_PROVIDER_MARKUP,
            default=defaults.get(CONF_PROVIDER_MARKUP, DEFAULT_PROVIDER_MARKUP),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                max=20,
                step=0.01,
                unit_of_measurement="ct/kWh",
                mode=selector.NumberSelectorMode.BOX,
            ),
        ),
        vol.Required(
            CONF_MAX_PRICE,
            default=defaults.get(CONF_MAX_PRICE, DEFAULT_MAX_PRICE),
        ): selector.NumberSelector(
            selector.NumberSelectorConfig(
                min=0,
                max=100,
                step=0.5,
                unit_of_measurement="ct/kWh",
                mode=selector.NumberSelectorMode.BOX,
            ),
        ),
        vol.Optional(
            CONF_BATTERY_POWER_SENSOR,
            default=defaults.get(CONF_BATTERY_POWER_SENSOR, ""),
        ): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="power",
                multiple=False,
            ),
        ),
    })


# ============================================================================
# CONFIG FLOW CLASS
# ============================================================================


class GridPriceMonitorConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the configuration flow for Grid Price Monitor @zara"""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow @zara"""
        self._calibration_spot_price: float | None = None
        self._selected_country: str = DEFAULT_COUNTRY
        self._selected_vat_rate: int = VAT_RATE_DE
        self._existing_data: dict[str, Any] = {}

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> GridPriceMonitorOptionsFlow:
        """Get the options flow handler @zara"""
        return GridPriceMonitorOptionsFlow()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial setup step - country selection @zara"""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._selected_country = user_input.get(CONF_COUNTRY, DEFAULT_COUNTRY)
            self._selected_vat_rate = _get_default_vat_for_country(self._selected_country)

            # Check if already configured
            await self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()

            return await self.async_step_pricing()

        return self.async_show_form(
            step_id="user",
            data_schema=_get_country_schema(),
            errors=errors,
        )

    async def async_step_pricing(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the pricing configuration step @zara"""
        errors: dict[str, str] = {}
        is_reconfigure = self.source == SOURCE_RECONFIGURE

        if user_input is not None:
            # Get VAT rate from input (comes as string from selector)
            vat_rate_str = user_input.get(CONF_VAT_RATE, str(self._selected_vat_rate))
            vat_rate = int(vat_rate_str)

            use_calibration = user_input.get(CONF_USE_CALIBRATION, False)
            grid_fee = user_input.get(CONF_GRID_FEE, DEFAULT_GRID_FEE)
            taxes_fees = user_input.get(CONF_TAXES_FEES, DEFAULT_TAXES_FEES)
            provider_markup = user_input.get(CONF_PROVIDER_MARKUP, DEFAULT_PROVIDER_MARKUP)

            if use_calibration:
                calibration_price = user_input.get(CONF_CALIBRATION_PRICE)
                if calibration_price is not None and self._calibration_spot_price is not None:
                    # Calculate total markup from calibration
                    # Calibration price is gross, spot price is net
                    # Total = (Spot × VAT) + Markup
                    # Markup = Total - (Spot × VAT)
                    vat_factor = 1 + (vat_rate / 100)
                    spot_gross = self._calibration_spot_price * vat_factor
                    total_markup = calibration_price - spot_gross

                    if total_markup > 0:
                        # Split roughly: 60% grid fee, 30% taxes, 10% provider
                        grid_fee = round(total_markup * 0.6, 2)
                        taxes_fees = round(total_markup * 0.3, 2)
                        provider_markup = round(total_markup * 0.1, 2)
                    else:
                        errors["base"] = "calibration_failed"
                else:
                    errors["base"] = "calibration_failed"

            if not errors:
                max_price = user_input.get(CONF_MAX_PRICE, DEFAULT_MAX_PRICE)
                battery_sensor = user_input.get(CONF_BATTERY_POWER_SENSOR, "")

                data = {
                    CONF_COUNTRY: self._selected_country,
                    CONF_VAT_RATE: vat_rate,
                    CONF_GRID_FEE: grid_fee,
                    CONF_TAXES_FEES: taxes_fees,
                    CONF_PROVIDER_MARKUP: provider_markup,
                    CONF_MAX_PRICE: max_price,
                    CONF_BATTERY_POWER_SENSOR: battery_sensor,
                }

                if is_reconfigure:
                    return self.async_update_reload_and_abort(
                        self._get_reconfigure_entry(),
                        data_updates=data,
                    )

                return self.async_create_entry(
                    title=NAME,
                    data=data,
                )

        # Fetch current spot price for calibration display
        if self._calibration_spot_price is None:
            try:
                # Lazy import to avoid blocking the event loop
                from .core import ElectricityPriceService

                service = ElectricityPriceService(self._selected_country)
                await service.fetch_day_ahead_prices()
                self._calibration_spot_price = service.get_current_price()
            except Exception as err:
                _LOGGER.warning("Could not fetch spot price for calibration: %s", err)
                self._calibration_spot_price = None

        description_placeholders = {}
        if self._calibration_spot_price is not None:
            # Show gross price (with default VAT) for user reference
            vat_factor = 1 + (self._selected_vat_rate / 100)
            spot_gross = self._calibration_spot_price * vat_factor
            description_placeholders["spot_price"] = f"{spot_gross:.2f}"
        else:
            description_placeholders["spot_price"] = "N/A"

        # Use existing data as defaults for reconfigure
        defaults = self._existing_data if is_reconfigure else None
        default_vat = (
            self._existing_data.get(CONF_VAT_RATE, self._selected_vat_rate)
            if is_reconfigure
            else self._selected_vat_rate
        )

        return self.async_show_form(
            step_id="pricing",
            data_schema=_get_pricing_schema(defaults, default_vat),
            errors=errors,
            description_placeholders=description_placeholders,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle reconfiguration - allows changing all settings including country @zara"""
        errors: dict[str, str] = {}
        entry = self._get_reconfigure_entry()
        self._existing_data = dict(entry.data)

        if user_input is not None:
            self._selected_country = user_input.get(
                CONF_COUNTRY,
                self._existing_data.get(CONF_COUNTRY, DEFAULT_COUNTRY)
            )
            # Update default VAT if country changes
            if self._selected_country != self._existing_data.get(CONF_COUNTRY):
                self._selected_vat_rate = _get_default_vat_for_country(self._selected_country)
                self._calibration_spot_price = None
            else:
                self._selected_vat_rate = self._existing_data.get(
                    CONF_VAT_RATE, _get_default_vat_for_country(self._selected_country)
                )

            return await self.async_step_pricing()

        current_country = self._existing_data.get(CONF_COUNTRY, DEFAULT_COUNTRY)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=_get_country_schema(current_country),
            errors=errors,
        )


# ============================================================================
# OPTIONS FLOW CLASS
# ============================================================================


class GridPriceMonitorOptionsFlow(config_entries.OptionsFlow):
    """Handle options flow for Grid Price Monitor @zara

    Options flow allows quick changes to pricing without changing country.
    For country changes, use the reconfigure flow.
    """

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options @zara"""
        errors: dict[str, str] = {}
        current_data = self.config_entry.data

        if user_input is not None:
            # Validate max_price
            max_price = user_input.get(CONF_MAX_PRICE, DEFAULT_MAX_PRICE)
            if max_price <= 0:
                errors[CONF_MAX_PRICE] = "invalid_max_price"

            # Get VAT rate from input (comes as string from selector)
            vat_rate_str = user_input.get(
                CONF_VAT_RATE,
                str(current_data.get(CONF_VAT_RATE, VAT_RATE_DE))
            )
            vat_rate = int(vat_rate_str)

            if not errors:
                # Merge with existing data (keep country)
                new_data = {
                    **self.config_entry.data,
                    CONF_VAT_RATE: vat_rate,
                    CONF_GRID_FEE: user_input.get(CONF_GRID_FEE),
                    CONF_TAXES_FEES: user_input.get(CONF_TAXES_FEES),
                    CONF_PROVIDER_MARKUP: user_input.get(CONF_PROVIDER_MARKUP),
                    CONF_MAX_PRICE: max_price,
                    CONF_BATTERY_POWER_SENSOR: user_input.get(CONF_BATTERY_POWER_SENSOR, ""),
                }

                self.hass.config_entries.async_update_entry(
                    self.config_entry,
                    data=new_data,
                )

                return self.async_create_entry(title="", data={})

        current_country = current_data.get(CONF_COUNTRY, DEFAULT_COUNTRY)
        current_vat = current_data.get(CONF_VAT_RATE, _get_default_vat_for_country(current_country))

        schema = vol.Schema({
            vol.Required(
                CONF_VAT_RATE,
                default=str(current_vat),
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        selector.SelectOptionDict(value=str(opt["value"]), label=opt["label"])
                        for opt in VAT_OPTIONS
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN,
                ),
            ),
            vol.Required(
                CONF_GRID_FEE,
                default=current_data.get(CONF_GRID_FEE, DEFAULT_GRID_FEE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=50,
                    step=0.01,
                    unit_of_measurement="ct/kWh",
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Required(
                CONF_TAXES_FEES,
                default=current_data.get(CONF_TAXES_FEES, DEFAULT_TAXES_FEES),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=50,
                    step=0.01,
                    unit_of_measurement="ct/kWh",
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Required(
                CONF_PROVIDER_MARKUP,
                default=current_data.get(CONF_PROVIDER_MARKUP, DEFAULT_PROVIDER_MARKUP),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=20,
                    step=0.01,
                    unit_of_measurement="ct/kWh",
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Required(
                CONF_MAX_PRICE,
                default=current_data.get(CONF_MAX_PRICE, DEFAULT_MAX_PRICE),
            ): selector.NumberSelector(
                selector.NumberSelectorConfig(
                    min=0,
                    max=100,
                    step=0.5,
                    unit_of_measurement="ct/kWh",
                    mode=selector.NumberSelectorMode.BOX,
                ),
            ),
            vol.Optional(
                CONF_BATTERY_POWER_SENSOR,
                default=current_data.get(CONF_BATTERY_POWER_SENSOR, ""),
            ): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="power",
                    multiple=False,
                ),
            ),
        })

        return self.async_show_form(
            step_id="init",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "country": COUNTRY_OPTIONS.get(current_country, current_country),
            },
        )
