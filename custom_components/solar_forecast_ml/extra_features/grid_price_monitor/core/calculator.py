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

_LOGGER = logging.getLogger(__name__)


class PriceCalculator:
    """Handles price calculations for electricity @zara"""

    def __init__(
        self,
        vat_rate: int = 19,
        grid_fee: float = 0.0,
        taxes_fees: float = 0.0,
        provider_markup: float = 0.0,
    ) -> None:
        """Initialize the price calculator @zara

        Args:
            vat_rate: VAT rate in percent (e.g., 19 for 19%)
            grid_fee: Grid fee in ct/kWh (gross)
            taxes_fees: Taxes and fees in ct/kWh (gross)
            provider_markup: Provider markup in ct/kWh (gross)
        """
        self._vat_rate = vat_rate
        self._grid_fee = grid_fee
        self._taxes_fees = taxes_fees
        self._provider_markup = provider_markup

    @property
    def vat_rate(self) -> int:
        """Get VAT rate @zara"""
        return self._vat_rate

    @vat_rate.setter
    def vat_rate(self, value: int) -> None:
        """Set VAT rate @zara"""
        self._vat_rate = value

    @property
    def vat_factor(self) -> float:
        """Get VAT factor (e.g., 1.19 for 19%) @zara"""
        return 1 + (self._vat_rate / 100)

    @property
    def grid_fee(self) -> float:
        """Get grid fee @zara"""
        return self._grid_fee

    @grid_fee.setter
    def grid_fee(self, value: float) -> None:
        """Set grid fee @zara"""
        self._grid_fee = value

    @property
    def taxes_fees(self) -> float:
        """Get taxes and fees @zara"""
        return self._taxes_fees

    @taxes_fees.setter
    def taxes_fees(self, value: float) -> None:
        """Set taxes and fees @zara"""
        self._taxes_fees = value

    @property
    def provider_markup(self) -> float:
        """Get provider markup @zara"""
        return self._provider_markup

    @provider_markup.setter
    def provider_markup(self, value: float) -> None:
        """Set provider markup @zara"""
        self._provider_markup = value

    @property
    def total_markup(self) -> float:
        """Calculate total markup (grid fee + taxes + provider) @zara"""
        return self._grid_fee + self._taxes_fees + self._provider_markup

    def calculate_gross_spot(self, spot_price_net: float) -> float:
        """Calculate gross spot price from net @zara

        Args:
            spot_price_net: Net spot price in ct/kWh

        Returns:
            Gross spot price in ct/kWh
        """
        return round(spot_price_net * self.vat_factor, 2)

    def calculate_total_price(self, spot_price_net: float) -> float:
        """Calculate total price from net spot price @zara

        Formula: (Spot_net × VAT_factor) + Grid_fee + Taxes + Provider_markup

        Args:
            spot_price_net: Net spot price in ct/kWh (from aWATTar)

        Returns:
            Total gross price in ct/kWh
        """
        spot_gross = spot_price_net * self.vat_factor
        return round(spot_gross + self.total_markup, 2)

    def calculate_spot_from_total(self, total_price: float) -> float:
        """Reverse calculate: get net spot price from total gross price @zara

        Used for calibration and threshold calculations.

        Args:
            total_price: Total gross price in ct/kWh

        Returns:
            Net spot price in ct/kWh
        """
        spot_gross = total_price - self.total_markup
        return round(spot_gross / self.vat_factor, 2)

    def calculate_markup_from_calibration(
        self,
        current_total_price: float,
        current_spot_net: float,
        vat_rate: int | None = None
    ) -> dict[str, float] | None:
        """Calculate markup components from calibration price @zara

        Args:
            current_total_price: Current total price from provider
            current_spot_net: Current net spot price
            vat_rate: Optional VAT rate override

        Returns:
            Dictionary with grid_fee, taxes_fees, provider_markup or None if invalid
        """
        if vat_rate is None:
            vat_rate = self._vat_rate

        vat_factor = 1 + (vat_rate / 100)
        spot_gross = current_spot_net * vat_factor
        total_markup = current_total_price - spot_gross

        if total_markup <= 0:
            _LOGGER.warning(
                "Calibration failed: negative markup (total=%.2f, spot_gross=%.2f)",
                current_total_price,
                spot_gross,
            )
            return None

        # Split markup roughly: 60% grid fee, 30% taxes, 10% provider
        return {
            "grid_fee": round(total_markup * 0.6, 2),
            "taxes_fees": round(total_markup * 0.3, 2),
            "provider_markup": round(total_markup * 0.1, 2),
        }

    def is_cheap(self, total_price: float, max_price: float) -> bool:
        """Check if the price is below the cheap threshold @zara

        Args:
            total_price: Total price in ct/kWh
            max_price: Maximum price threshold

        Returns:
            True if price is below threshold
        """
        return total_price < max_price

    def calculate_trend(
        self,
        current_price: float | None,
        next_price: float | None
    ) -> str:
        """Calculate price trend @zara

        Args:
            current_price: Current total price
            next_price: Next hour total price

        Returns:
            Trend string: "rising", "falling", "stable", or "unknown"
        """
        if current_price is None or next_price is None:
            return "unknown"

        diff = next_price - current_price

        if diff > 1:
            return "rising"
        elif diff < -1:
            return "falling"
        else:
            return "stable"

    def build_forecast_entry(
        self,
        price_entry: dict[str, Any],
        max_price: float
    ) -> dict[str, Any]:
        """Build a forecast entry with calculated prices @zara

        Args:
            price_entry: Raw price entry with 'price' and 'hour'
            max_price: Maximum price threshold

        Returns:
            Enriched forecast entry
        """
        spot_net = price_entry.get("price", 0)
        spot_gross = self.calculate_gross_spot(spot_net)
        total = self.calculate_total_price(spot_net)

        return {
            "hour": price_entry.get("hour"),
            "timestamp": price_entry.get("timestamp"),
            "spot_price_net": spot_net,
            "spot_price": spot_gross,
            "total_price": total,
            "is_cheap": self.is_cheap(total, max_price),
        }

    def update_config(
        self,
        vat_rate: int | None = None,
        grid_fee: float | None = None,
        taxes_fees: float | None = None,
        provider_markup: float | None = None,
    ) -> None:
        """Update calculator configuration @zara

        Args:
            vat_rate: New VAT rate
            grid_fee: New grid fee
            taxes_fees: New taxes and fees
            provider_markup: New provider markup
        """
        if vat_rate is not None:
            self._vat_rate = vat_rate
        if grid_fee is not None:
            self._grid_fee = grid_fee
        if taxes_fees is not None:
            self._taxes_fees = taxes_fees
        if provider_markup is not None:
            self._provider_markup = provider_markup

        _LOGGER.debug(
            "Calculator updated: VAT=%d%%, markup=%.2f ct/kWh",
            self._vat_rate,
            self.total_markup,
        )

    def get_config(self) -> dict[str, Any]:
        """Get current calculator configuration @zara

        Returns:
            Dictionary with current configuration
        """
        return {
            "vat_rate": self._vat_rate,
            "vat_factor": self.vat_factor,
            "grid_fee": self._grid_fee,
            "taxes_fees": self._taxes_fees,
            "provider_markup": self._provider_markup,
            "total_markup": self.total_markup,
        }
