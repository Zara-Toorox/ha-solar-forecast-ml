# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Monthly tariff manager for EEG and Energy Sharing support. @zara"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiofiles

from homeassistant.core import HomeAssistant

from ..const import (
    DOMAIN,
    SFML_STATS_DATA,
    MONTHLY_TARIFFS_FILE,
    HOURLY_BILLING_HISTORY,
    CONF_FEED_IN_TARIFF,
    CONF_BILLING_FIXED_PRICE,
    CONF_REFERENCE_PRICE,
    CONF_EEG_IMPORT_PRICE,
    CONF_EEG_FEED_IN_PRICE,
    CONF_GRID_FEE_BASE,
    CONF_GRID_FEE_SCALING,
    DEFAULT_FEED_IN_TARIFF,
    DEFAULT_BILLING_FIXED_PRICE,
    DEFAULT_REFERENCE_PRICE,
    DEFAULT_EEG_IMPORT_PRICE,
    DEFAULT_EEG_FEED_IN_PRICE,
    DEFAULT_GRID_FEE_BASE,
    DEFAULT_GRID_FEE_SCALING,
    GRID_FEE_THRESHOLD_HIGH,
    GRID_FEE_THRESHOLD_MEDIUM,
    GRID_FEE_THRESHOLD_LOW,
    GRID_FEE_FACTOR_HIGH,
    GRID_FEE_FACTOR_MEDIUM,
    GRID_FEE_FACTOR_LOW,
    GRID_FEE_FACTOR_VERY_LOW,
)

_LOGGER = logging.getLogger(__name__)


class MonthlyTariffManager:
    """Manage monthly tariffs with smart defaults and manual overrides. @zara"""

    def __init__(
        self,
        hass: HomeAssistant,
        config_path: Path,
        entry_data: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the tariff manager. @zara"""
        self._hass = hass
        self._config_path = config_path
        self._entry_data = entry_data
        self._data_path = config_path / SFML_STATS_DATA
        self._tariff_file = self._data_path / MONTHLY_TARIFFS_FILE
        self._hourly_file = self._data_path / HOURLY_BILLING_HISTORY
        self._cache: dict[str, Any] | None = None

    def update_config(self, new_config: dict[str, Any]) -> None:
        """Update cached configuration. @zara"""
        self._entry_data = new_config
        _LOGGER.debug("MonthlyTariffManager config updated")

    def _get_config(self) -> dict[str, Any]:
        """Get current configuration. @zara"""
        if self._entry_data:
            return self._entry_data

        entries = self._hass.data.get(DOMAIN, {})
        for entry_id, entry_data in entries.items():
            if isinstance(entry_data, dict) and "config" in entry_data:
                return entry_data["config"]

        config_entries = self._hass.config_entries.async_entries(DOMAIN)
        if config_entries:
            return dict(config_entries[0].data)
        return {}

    def _get_defaults(self) -> dict[str, Any]:
        """Get default tariff values from configuration. @zara"""
        config = self._get_config()
        return {
            "reference_price_ct": config.get(
                CONF_REFERENCE_PRICE, DEFAULT_REFERENCE_PRICE
            ),
            "feed_in_tariff_ct": config.get(
                CONF_FEED_IN_TARIFF, DEFAULT_FEED_IN_TARIFF
            ),
            "fixed_price_ct": config.get(
                CONF_BILLING_FIXED_PRICE, DEFAULT_BILLING_FIXED_PRICE
            ),
            "eeg_import_price_ct": config.get(
                CONF_EEG_IMPORT_PRICE, DEFAULT_EEG_IMPORT_PRICE
            ),
            "eeg_feed_in_ct": config.get(
                CONF_EEG_FEED_IN_PRICE, DEFAULT_EEG_FEED_IN_PRICE
            ),
            "grid_fee_base_ct": config.get(
                CONF_GRID_FEE_BASE, DEFAULT_GRID_FEE_BASE
            ),
            "grid_fee_scaling_enabled": config.get(
                CONF_GRID_FEE_SCALING, DEFAULT_GRID_FEE_SCALING
            ),
        }

    async def _load_data(self) -> dict[str, Any]:
        """Load tariff data from file. @zara"""
        if self._cache is not None:
            return self._cache

        if not self._tariff_file.exists():
            self._cache = {
                "version": 1,
                "defaults": self._get_defaults(),
                "months": {},
            }
            return self._cache

        try:
            async with aiofiles.open(self._tariff_file, "r", encoding="utf-8") as f:
                content = await f.read()
                self._cache = json.loads(content)
                return self._cache
        except Exception as err:
            _LOGGER.error("Error loading tariff data: %s", err)
            self._cache = {
                "version": 1,
                "defaults": self._get_defaults(),
                "months": {},
            }
            return self._cache

    async def _save_data(self, data: dict[str, Any]) -> bool:
        """Save tariff data to file. @zara"""
        try:
            self._data_path.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(self._tariff_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
            self._cache = data
            return True
        except Exception as err:
            _LOGGER.error("Error saving tariff data: %s", err)
            return False

    async def _load_hourly_data(self) -> dict[str, Any]:
        """Load hourly billing history data. @zara"""
        if not self._hourly_file.exists():
            return {"hours": {}}

        try:
            async with aiofiles.open(self._hourly_file, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as err:
            _LOGGER.error("Error loading hourly data: %s", err)
            return {"hours": {}}

    def _get_month_key(self, year: int, month: int) -> str:
        """Generate month key in YYYY-MM format. @zara"""
        return f"{year:04d}-{month:02d}"

    def _calculate_grid_fee(self, annual_import_kwh: float) -> float:
        """Calculate scaled grid fee based on annual consumption. @zara"""
        defaults = self._get_defaults()
        base_fee = defaults["grid_fee_base_ct"]

        if not defaults["grid_fee_scaling_enabled"]:
            return base_fee

        if annual_import_kwh > GRID_FEE_THRESHOLD_HIGH:
            return base_fee * GRID_FEE_FACTOR_HIGH
        elif annual_import_kwh > GRID_FEE_THRESHOLD_MEDIUM:
            return base_fee * GRID_FEE_FACTOR_MEDIUM
        elif annual_import_kwh > GRID_FEE_THRESHOLD_LOW:
            return base_fee * GRID_FEE_FACTOR_LOW
        else:
            return base_fee * GRID_FEE_FACTOR_VERY_LOW

    def _estimate_eeg_share(
        self,
        weighted_price: float,
        standard_price: float,
        eeg_price: float,
    ) -> float:
        """Estimate EEG share percentage from price difference. @zara"""
        if standard_price <= eeg_price:
            return 0.0

        share = (standard_price - weighted_price) / (standard_price - eeg_price) * 100
        return max(0.0, min(100.0, share))

    async def calculate_weighted_average_price(
        self, year: int, month: int
    ) -> dict[str, float]:
        """Calculate consumption-weighted average price for a month. @zara"""
        month_key = self._get_month_key(year, month)
        hourly_data = await self._load_hourly_data()
        hours = hourly_data.get("hours", {})

        total_cost_ct = 0.0
        total_import_kwh = 0.0
        total_export_kwh = 0.0
        total_self_consumption_kwh = 0.0
        price_count = 0
        price_sum = 0.0

        for hour_key, hour_data in hours.items():
            if not hour_key.startswith(month_key):
                continue

            import_kwh = hour_data.get("grid_import_kwh", 0) or 0
            export_kwh = hour_data.get("grid_export_kwh", 0) or 0
            price_ct = hour_data.get("price_ct_kwh", 0) or 0
            solar_to_house = hour_data.get("solar_to_house_kwh", 0) or 0
            battery_to_house = hour_data.get("battery_to_house_kwh", 0) or 0

            if import_kwh > 0 and price_ct > 0:
                total_cost_ct += import_kwh * price_ct
                total_import_kwh += import_kwh

            total_export_kwh += export_kwh
            total_self_consumption_kwh += solar_to_house + battery_to_house

            if price_ct > 0:
                price_sum += price_ct
                price_count += 1

        if total_import_kwh > 0:
            weighted_avg_price = total_cost_ct / total_import_kwh
        elif price_count > 0:
            weighted_avg_price = price_sum / price_count
        else:
            weighted_avg_price = self._get_defaults()["fixed_price_ct"]

        return {
            "weighted_avg_price_ct": round(weighted_avg_price, 2),
            "import_kwh": round(total_import_kwh, 2),
            "export_kwh": round(total_export_kwh, 2),
            "self_consumption_kwh": round(total_self_consumption_kwh, 2),
            "hours_with_data": price_count,
        }

    async def get_monthly_data(self, year: int, month: int) -> dict[str, Any]:
        """Get complete monthly data with auto-calculated and override values. @zara"""
        month_key = self._get_month_key(year, month)
        data = await self._load_data()
        defaults = self._get_defaults()

        auto_calc = await self.calculate_weighted_average_price(year, month)

        eeg_share = None
        if auto_calc["weighted_avg_price_ct"] > 0:
            eeg_share = self._estimate_eeg_share(
                auto_calc["weighted_avg_price_ct"],
                defaults["reference_price_ct"],
                defaults["eeg_import_price_ct"],
            )

        monthly_import = auto_calc["import_kwh"]
        current_month = date.today().month
        months_elapsed = month - 1 + (12 if year < date.today().year else 0) + 1
        if months_elapsed > 0 and monthly_import > 0:
            projected_annual = (monthly_import / months_elapsed) * 12
        else:
            projected_annual = 3000
        estimated_grid_fee = self._calculate_grid_fee(projected_annual)

        month_data = data.get("months", {}).get(month_key, {})
        overrides = month_data.get("overrides", {})
        is_finalized = month_data.get("is_finalized", False)
        finalized_at = month_data.get("finalized_at")

        def get_effective(key: str, auto_value: float, default_value: float) -> dict:
            if key in overrides and overrides[key] is not None:
                return {
                    "value": overrides[key],
                    "source": "manual",
                }
            elif auto_value is not None and auto_value > 0:
                return {
                    "value": auto_value,
                    "source": "auto",
                }
            else:
                return {
                    "value": default_value,
                    "source": "default",
                }

        result = {
            "month_key": month_key,
            "year": year,
            "month": month,
            "is_finalized": is_finalized,
            "finalized_at": finalized_at,
            "auto_calculated": {
                "import_kwh": auto_calc["import_kwh"],
                "export_kwh": auto_calc["export_kwh"],
                "self_consumption_kwh": auto_calc["self_consumption_kwh"],
                "weighted_avg_price_ct": auto_calc["weighted_avg_price_ct"],
                "eeg_share_percent": round(eeg_share, 1) if eeg_share else None,
                "estimated_grid_fee_ct": round(estimated_grid_fee, 2),
                "hours_with_data": auto_calc["hours_with_data"],
            },
            "overrides": overrides,
            "effective": {
                "import_price_ct": get_effective(
                    "import_price_ct",
                    auto_calc["weighted_avg_price_ct"],
                    defaults["fixed_price_ct"],
                ),
                "export_price_ct": get_effective(
                    "export_price_ct",
                    None,
                    defaults["feed_in_tariff_ct"],
                ),
                "reference_price_ct": get_effective(
                    "reference_price_ct",
                    None,
                    defaults["reference_price_ct"],
                ),
                "grid_fee_ct": get_effective(
                    "grid_fee_ct",
                    estimated_grid_fee,
                    defaults["grid_fee_base_ct"],
                ),
                "eeg_share_percent": get_effective(
                    "eeg_share_percent",
                    eeg_share,
                    0.0,
                ),
            },
            "defaults": defaults,
        }

        return result

    async def set_monthly_override(
        self,
        year: int,
        month: int,
        overrides: dict[str, float | None],
    ) -> bool:
        """Set manual override values for a month. @zara"""
        month_key = self._get_month_key(year, month)
        data = await self._load_data()

        if "months" not in data:
            data["months"] = {}

        if month_key not in data["months"]:
            data["months"][month_key] = {
                "overrides": {},
                "is_finalized": False,
            }

        existing_overrides = data["months"][month_key].get("overrides", {})
        for key, value in overrides.items():
            if value is None:
                existing_overrides.pop(key, None)
            else:
                existing_overrides[key] = value

        data["months"][month_key]["overrides"] = existing_overrides
        data["months"][month_key]["updated_at"] = datetime.now().isoformat()

        return await self._save_data(data)

    async def finalize_month(
        self,
        year: int,
        month: int,
        recalculate_history: bool = True,
    ) -> dict[str, Any]:
        """Mark a month as finalized after billing. @zara"""
        month_key = self._get_month_key(year, month)
        data = await self._load_data()

        if "months" not in data:
            data["months"] = {}

        if month_key not in data["months"]:
            data["months"][month_key] = {"overrides": {}}

        data["months"][month_key]["is_finalized"] = True
        data["months"][month_key]["finalized_at"] = datetime.now().isoformat()

        await self._save_data(data)

        result = {
            "success": True,
            "month_key": month_key,
            "finalized_at": data["months"][month_key]["finalized_at"],
            "recalculated": False,
        }

        if recalculate_history:
            recalc_result = await self._recalculate_month_history(year, month)
            result["recalculated"] = recalc_result
            result["recalculation_details"] = recalc_result

        return result

    async def unfinalize_month(self, year: int, month: int) -> bool:
        """Remove finalization status from a month. @zara"""
        month_key = self._get_month_key(year, month)
        data = await self._load_data()

        if month_key in data.get("months", {}):
            data["months"][month_key]["is_finalized"] = False
            data["months"][month_key].pop("finalized_at", None)
            return await self._save_data(data)

        return True

    async def _recalculate_month_history(
        self, year: int, month: int
    ) -> dict[str, Any]:
        """Recalculate historical data for a month with correct prices. @zara"""
        month_key = self._get_month_key(year, month)
        month_data = await self.get_monthly_data(year, month)

        effective = month_data["effective"]
        import_price = effective["import_price_ct"]["value"]
        export_price = effective["export_price_ct"]["value"]
        reference_price = effective["reference_price_ct"]["value"]

        daily_file = self._data_path / "daily_energy_history.json"
        if not daily_file.exists():
            return {"success": False, "error": "No daily history file"}

        try:
            async with aiofiles.open(daily_file, "r", encoding="utf-8") as f:
                content = await f.read()
                daily_data = json.loads(content)
        except Exception as err:
            return {"success": False, "error": str(err)}

        days_updated = 0
        days = daily_data.get("days", {})

        for day_key, day_data in days.items():
            if not day_key.startswith(month_key):
                continue

            import_kwh = day_data.get("grid_import_kwh", 0) or 0
            export_kwh = day_data.get("grid_export_kwh", 0) or 0
            self_consumption_kwh = (
                (day_data.get("solar_to_house_kwh", 0) or 0) +
                (day_data.get("battery_to_house_kwh", 0) or 0)
            )

            day_data["finalized_import_price_ct"] = import_price
            day_data["finalized_export_price_ct"] = export_price
            day_data["finalized_reference_price_ct"] = reference_price
            day_data["grid_cost_eur"] = round((import_kwh * import_price) / 100, 2)
            day_data["feed_in_revenue_eur"] = round((export_kwh * export_price) / 100, 2)
            day_data["savings_eur"] = round((self_consumption_kwh * reference_price) / 100, 2)
            day_data["is_finalized"] = True

            days_updated += 1

        daily_data["days"] = days
        daily_data["last_recalculation"] = datetime.now().isoformat()

        try:
            async with aiofiles.open(daily_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(daily_data, indent=2, ensure_ascii=False))
        except Exception as err:
            return {"success": False, "error": str(err)}

        _LOGGER.info(
            "Recalculated %d days for %s with import=%.2f, export=%.2f, ref=%.2f ct/kWh",
            days_updated, month_key, import_price, export_price, reference_price
        )

        return {
            "success": True,
            "days_updated": days_updated,
            "prices_used": {
                "import_ct": import_price,
                "export_ct": export_price,
                "reference_ct": reference_price,
            },
        }

    async def get_all_months(
        self,
        year: int | None = None,
        include_empty: bool = False,
    ) -> list[dict[str, Any]]:
        """Get data for all months, optionally filtered by year. @zara"""
        if year is None:
            year = date.today().year

        months = []
        for month in range(1, 13):
            if year == date.today().year and month > date.today().month:
                if not include_empty:
                    continue

            month_data = await self.get_monthly_data(year, month)
            months.append(month_data)

        return months

    async def get_year_summary(self, year: int) -> dict[str, Any]:
        """Get yearly summary with totals and averages. @zara"""
        months = await self.get_all_months(year)

        total_import_kwh = 0.0
        total_export_kwh = 0.0
        total_self_consumption_kwh = 0.0
        total_grid_cost = 0.0
        total_feed_in_revenue = 0.0
        total_savings = 0.0
        finalized_months = 0
        price_sum = 0.0
        price_count = 0

        for m in months:
            auto = m["auto_calculated"]
            eff = m["effective"]

            total_import_kwh += auto["import_kwh"]
            total_export_kwh += auto["export_kwh"]
            total_self_consumption_kwh += auto["self_consumption_kwh"]

            import_price = eff["import_price_ct"]["value"]
            export_price = eff["export_price_ct"]["value"]
            reference_price = eff["reference_price_ct"]["value"]

            total_grid_cost += (auto["import_kwh"] * import_price) / 100
            total_feed_in_revenue += (auto["export_kwh"] * export_price) / 100
            total_savings += (auto["self_consumption_kwh"] * reference_price) / 100

            if m["is_finalized"]:
                finalized_months += 1

            if import_price > 0:
                price_sum += import_price
                price_count += 1

        avg_import_price = price_sum / price_count if price_count > 0 else 0

        return {
            "year": year,
            "months_with_data": len([m for m in months if m["auto_calculated"]["hours_with_data"] > 0]),
            "finalized_months": finalized_months,
            "totals": {
                "import_kwh": round(total_import_kwh, 2),
                "export_kwh": round(total_export_kwh, 2),
                "self_consumption_kwh": round(total_self_consumption_kwh, 2),
                "grid_cost_eur": round(total_grid_cost, 2),
                "feed_in_revenue_eur": round(total_feed_in_revenue, 2),
                "savings_eur": round(total_savings, 2),
                "net_benefit_eur": round(total_savings + total_feed_in_revenue - total_grid_cost, 2),
            },
            "averages": {
                "import_price_ct": round(avg_import_price, 2),
            },
            "months": months,
        }

    async def export_csv(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> str:
        """Export monthly data as CSV. @zara"""
        lines = [
            "Monat;Bezug (kWh);Bezugspreis (ct/kWh);Quelle Bezugspreis;"
            "Einspeisung (kWh);Vergütung (ct/kWh);Quelle Vergütung;"
            "Eigenverbrauch (kWh);Referenzpreis (ct/kWh);"
            "Netzgebühren (ct/kWh);EEG-Anteil (%);"
            "Stromkosten (EUR);Einspeise-Erlös (EUR);Einsparung (EUR);"
            "Status"
        ]

        current = date(start_year, start_month, 1)
        end = date(end_year, end_month, 1)

        while current <= end:
            m = await self.get_monthly_data(current.year, current.month)
            auto = m["auto_calculated"]
            eff = m["effective"]

            import_price = eff["import_price_ct"]["value"]
            export_price = eff["export_price_ct"]["value"]
            reference_price = eff["reference_price_ct"]["value"]
            grid_fee = eff["grid_fee_ct"]["value"]
            eeg_share = eff["eeg_share_percent"]["value"]

            grid_cost = (auto["import_kwh"] * import_price) / 100
            feed_in = (auto["export_kwh"] * export_price) / 100
            savings = (auto["self_consumption_kwh"] * reference_price) / 100

            status = "Finalisiert" if m["is_finalized"] else "Offen"

            line = (
                f"{m['month_key']};"
                f"{auto['import_kwh']:.2f};"
                f"{import_price:.2f};{eff['import_price_ct']['source']};"
                f"{auto['export_kwh']:.2f};"
                f"{export_price:.2f};{eff['export_price_ct']['source']};"
                f"{auto['self_consumption_kwh']:.2f};"
                f"{reference_price:.2f};"
                f"{grid_fee:.2f};"
                f"{eeg_share:.1f};"
                f"{grid_cost:.2f};"
                f"{feed_in:.2f};"
                f"{savings:.2f};"
                f"{status}"
            )
            lines.append(line)

            if current.month == 12:
                current = date(current.year + 1, 1, 1)
            else:
                current = date(current.year, current.month + 1, 1)

        return "\n".join(lines)

    async def update_defaults(self, defaults: dict[str, Any]) -> bool:
        """Update default tariff values. @zara"""
        data = await self._load_data()
        data["defaults"] = {**data.get("defaults", {}), **defaults}
        return await self._save_data(data)

    def invalidate_cache(self) -> None:
        """Invalidate the data cache. @zara"""
        self._cache = None
