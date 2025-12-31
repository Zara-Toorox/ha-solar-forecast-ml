# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from homeassistant.core import HomeAssistant

from ..const import DATA_VERSION
from ..core.core_helpers import SafeDateTimeUtil as dt_util
from .data_io import DataManagerIO

_LOGGER = logging.getLogger(__name__)

class DataForecastHandler(DataManagerIO):
    """Handles daily forecasts TODAY block statistics and history"""

    def __init__(self, hass: HomeAssistant, data_dir: Path):
        super().__init__(hass, data_dir)

        self.daily_forecasts_file = self.data_dir / "stats" / "daily_forecasts.json"

        self._daily_forecasts_default = {
            "version": DATA_VERSION,
            "today": {
                "date": None,
                "forecast_day": {
                    "prediction_kwh": None,
                    "locked": False,
                    "locked_at": None,
                    "source": None,
                },
                "forecast_tomorrow": {
                    "date": None,
                    "prediction_kwh": None,
                    "locked": False,
                    "locked_at": None,
                    "source": None,
                    "updates": [],
                },
                "forecast_day_after_tomorrow": {
                    "date": None,
                    "prediction_kwh": None,
                    "locked": False,
                    "next_update": None,
                    "source": None,
                    "updates": [],
                },
                "forecast_best_hour": {
                    "hour": None,
                    "prediction_kwh": None,
                    "locked": False,
                    "locked_at": None,
                    "source": None,
                },
                "actual_best_hour": {"hour": None, "actual_kwh": None, "saved_at": None},
                "forecast_next_hour": {
                    "period": None,
                    "prediction_kwh": None,
                    "updated_at": None,
                    "source": None,
                },
                "production_time": {
                    "active": False,
                    "duration_seconds": 0,
                    "start_time": None,
                    "end_time": None,
                    "last_power_above_10w": None,
                    "zero_power_since": None,
                },
                "peak_today": {"power_w": 0.0, "at": None},
                "yield_today": {"kwh": None, "sensor": None},
                "consumption_today": {"kwh": None, "sensor": None},
                "autarky": {"percent": None, "calculated_at": None},
                "finalized": {
                    "yield_kwh": None,
                    "consumption_kwh": None,
                    "production_hours": None,
                    "accuracy_percent": None,
                    "at": None,
                },
            },
            "statistics": {
                "all_time_peak": {"power_w": 0.0, "date": None, "at": None},
                "current_week": {
                    "period": None,
                    "date_range": None,
                    "yield_kwh": 0.0,
                    "consumption_kwh": 0.0,
                    "days": 0,
                    "updated_at": None,
                },
                "current_month": {
                    "period": None,
                    "yield_kwh": 0.0,
                    "consumption_kwh": 0.0,
                    "avg_autarky": 0.0,
                    "days": 0,
                    "updated_at": None,
                },
                "last_7_days": {
                    "avg_yield_kwh": 0.0,
                    "avg_accuracy": 0.0,
                    "total_yield_kwh": 0.0,
                    "calculated_at": None,
                },
                "last_30_days": {
                    "avg_yield_kwh": 0.0,
                    "avg_accuracy": 0.0,
                    "total_yield_kwh": 0.0,
                    "calculated_at": None,
                },
                "last_365_days": {
                    "avg_yield_kwh": 0.0,
                    "total_yield_kwh": 0.0,
                    "calculated_at": None,
                },
            },
            "history": [],
            "metadata": {"retention_days": 730, "history_entries": 0, "last_update": None},
        }

    async def load_daily_forecasts(self) -> Dict[str, Any]:
        """Load daily forecasts file and ensure all required fields exist @zara"""
        data = await self._read_json_file(self.daily_forecasts_file, self._daily_forecasts_default)

        if "today" in data:
            needs_save = False
            today_default = self._daily_forecasts_default["today"]

            for field_name, field_default in today_default.items():
                if field_name not in data["today"]:
                    _LOGGER.warning(f"Missing field 'today.{field_name}' - adding default value")
                    data["today"][field_name] = (
                        field_default.copy() if isinstance(field_default, dict) else field_default
                    )
                    needs_save = True

            if needs_save:
                await self._atomic_write_json(self.daily_forecasts_file, data)
                _LOGGER.info("Auto-repaired daily_forecasts.json with missing fields")

        return data

    async def reset_today_block(self) -> bool:
        """Reset TODAY block at midnight @zara"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()
            today_str = now_local.date().isoformat()

            if "today" in data and data["today"].get("finalized"):
                await self._archive_yesterday(data["today"])

            data["today"] = self._daily_forecasts_default["today"].copy()
            data["today"]["date"] = today_str

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(f"TODAY block reset for {today_str}")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to reset TODAY block: {e}", exc_info=True)
            return False

    async def _archive_yesterday(self, yesterday_data: dict) -> None:
        """Archive completed day to history @zara"""
        try:
            finalized = yesterday_data.get("finalized", {})
            if not finalized.get("yield_kwh"):
                return

            history_entry = {
                "date": yesterday_data.get("date"),
                "predicted_kwh": yesterday_data.get("forecast_day", {}).get("prediction_kwh"),
                "actual_kwh": finalized.get("yield_kwh"),
                "consumption_kwh": finalized.get("consumption_kwh"),
                "autarky": yesterday_data.get("autarky", {}).get("percent"),
                "accuracy": finalized.get("accuracy_percent"),
                "production_hours": finalized.get("production_hours"),
                "peak_power": yesterday_data.get("peak_today", {}).get("power_w"),
                "source": yesterday_data.get("forecast_day", {}).get("source"),
                "archived_at": dt_util.now().isoformat(),
            }

            data = await self.load_daily_forecasts()
            data["history"].insert(0, history_entry)

            if len(data["history"]) > 730:
                data["history"] = data["history"][:730]

            data["metadata"]["history_entries"] = len(data["history"])

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(f" Yesterday archived: {yesterday_data.get('date')}")

        except Exception as e:
            _LOGGER.error(f"Failed to archive yesterday: {e}", exc_info=True)

    async def save_forecast_day(
        self,
        prediction_kwh: float,
        source: str = "ML",
        lock: bool = True,
        force_overwrite: bool = False,
        prediction_kwh_raw: float = None,
        safeguard_applied: bool = False,
    ) -> bool:
        """Save todays daily forecast locked at 6 AM"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()
            today_str = now_local.date().isoformat()

            if "today" not in data or data["today"].get("date") != today_str:
                data["today"] = self._daily_forecasts_default["today"].copy()
                data["today"]["date"] = today_str

            if lock and data["today"]["forecast_day"].get("locked") and not force_overwrite:
                _LOGGER.warning("Today's forecast already locked, skipping update")
                return False

            if force_overwrite and data["today"]["forecast_day"].get("locked"):
                old_value = data["today"]["forecast_day"].get("prediction_kwh")
                old_source = data["today"]["forecast_day"].get("source")
                _LOGGER.debug(
                    f"Force overwrite: Replacing locked forecast "
                    f"(old: {old_value} kWh from {old_source}) with new value "
                    f"(new: {prediction_kwh:.2f} kWh from {source})"
                )

            data["today"]["forecast_day"] = {
                "prediction_kwh": round(float(prediction_kwh), 2),
                "prediction_kwh_raw": round(float(prediction_kwh_raw), 2) if prediction_kwh_raw is not None else None,
                "safeguard_applied": safeguard_applied,
                "safeguard_reduction_kwh": round(float(prediction_kwh_raw - prediction_kwh), 2) if prediction_kwh_raw is not None else 0.0,
                "locked": lock,
                "locked_at": now_local.isoformat() if lock else None,
                "source": source,
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            lock_str = "[LOCKED]" if lock else "[UNLOCKED]"
            _LOGGER.info(
                f"✓ Today's forecast saved: {prediction_kwh:.2f} kWh "
                f"(source: {source}) {lock_str}"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save today's forecast: {e}", exc_info=True)
            return False

    async def save_forecast_tomorrow(
        self, date: datetime, prediction_kwh: float, source: str = "ML", lock: bool = False
    ) -> bool:
        """Save tomorrows forecast"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()
            date_str = date.date().isoformat()

            if "today" not in data:
                data["today"] = self._daily_forecasts_default["today"].copy()

            if lock and data["today"]["forecast_tomorrow"].get("locked"):
                _LOGGER.warning("Tomorrow's forecast already locked")
                return False

            update_entry = {
                "prediction_kwh": round(float(prediction_kwh), 2),
                "updated_at": now_local.isoformat(),
                "source": source,
            }

            updates = data["today"]["forecast_tomorrow"].get("updates", [])
            updates.append(update_entry)

            data["today"]["forecast_tomorrow"] = {
                "date": date_str,
                "prediction_kwh": round(float(prediction_kwh), 2),
                "locked": lock,
                "locked_at": now_local.isoformat() if lock else None,
                "source": source,
                "updates": updates[-10:],
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(f"✓ Tomorrow forecast saved: {date_str} = {prediction_kwh:.2f} kWh")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save tomorrow forecast: {e}", exc_info=True)
            return False

    async def save_forecast_day_after(
        self, date: datetime, prediction_kwh: float, source: str = "ML", lock: bool = False
    ) -> bool:
        """Save day after tomorrows forecast"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()
            date_str = date.date().isoformat()

            if "today" not in data:
                data["today"] = self._daily_forecasts_default["today"].copy()

            if lock and data["today"]["forecast_day_after_tomorrow"].get("locked"):
                _LOGGER.warning("Day after tomorrow's forecast already locked")
                return False

            update_entry = {
                "prediction_kwh": round(float(prediction_kwh), 2),
                "updated_at": now_local.isoformat(),
                "source": source,
            }

            updates = data["today"]["forecast_day_after_tomorrow"].get("updates", [])
            updates.append(update_entry)

            data["today"]["forecast_day_after_tomorrow"] = {
                "date": date_str,
                "prediction_kwh": round(float(prediction_kwh), 2),
                "locked": lock,
                "next_update": None,
                "source": source,
                "updates": updates[-10:],
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(
                f"✓ Day after tomorrow forecast saved: {date_str} = {prediction_kwh:.2f} kWh"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save day after tomorrow forecast: {e}", exc_info=True)
            return False

    async def save_forecast_best_hour(
        self, hour: int, prediction_kwh: float, source: str = "ML_Hourly"
    ) -> bool:
        """Save best hour forecast locked at 6 AM"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()

            if "today" not in data:
                data["today"] = self._daily_forecasts_default["today"].copy()

            data["today"]["forecast_best_hour"] = {
                "hour": hour,
                "prediction_kwh": round(float(prediction_kwh), 2),
                "locked": True,
                "locked_at": now_local.isoformat(),
                "source": source,
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(
                f"✓ Best hour forecast saved: Hour {hour} with {prediction_kwh:.2f} kWh [LOCKED]"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save best hour forecast: {e}", exc_info=True)
            return False

    async def save_actual_best_hour(self, hour: int, actual_kwh: float) -> bool:
        """Save actual best production hour calculated from hourly_samples at end of day @zara"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()

            if "today" not in data:
                data["today"] = self._daily_forecasts_default["today"].copy()

            data["today"]["actual_best_hour"] = {
                "hour": hour,
                "actual_kwh": round(float(actual_kwh), 2),
                "saved_at": now_local.isoformat(),
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(f"✓ Actual best hour saved: Hour {hour} with {actual_kwh:.2f} kWh")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save actual best hour: {e}", exc_info=True)
            return False

    async def save_forecast_next_hour(
        self,
        hour_start: datetime,
        hour_end: datetime,
        prediction_kwh: float,
        source: str = "ML_Hourly",
    ) -> bool:
        """Save next hour forecast updated every hour"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()

            if "today" not in data:
                data["today"] = self._daily_forecasts_default["today"].copy()

            period = f"{hour_start.isoformat()}/{hour_end.isoformat()}"

            data["today"]["forecast_next_hour"] = {
                "period": period,
                "prediction_kwh": round(float(prediction_kwh), 2),
                "updated_at": now_local.isoformat(),
                "source": source,
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.debug(
                f"Next hour forecast saved: {hour_start.strftime('%H:%M')}-{hour_end.strftime('%H:%M')} "
                f"= {prediction_kwh:.2f} kWh"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save next hour forecast: {e}", exc_info=True)
            return False

    async def deactivate_next_hour_forecast(self) -> bool:
        """Deactivate next hour forecast set to None @zara"""
        try:
            data = await self.load_daily_forecasts()

            if "today" not in data:
                return True

            data["today"]["forecast_next_hour"] = {
                "period": None,
                "prediction_kwh": None,
                "updated_at": None,
                "source": None,
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)
            _LOGGER.debug("Next hour forecast deactivated")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to deactivate next hour forecast: {e}", exc_info=True)
            return False

    async def update_production_time(
        self,
        active: bool,
        duration_seconds: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        last_power_above_10w: Optional[datetime] = None,
        zero_power_since: Optional[datetime] = None,
    ) -> bool:
        """Update production time tracking LIVE - every 30s"""
        try:
            data = await self.load_daily_forecasts()

            if "today" not in data:
                data["today"] = self._daily_forecasts_default["today"].copy()

            data["today"]["production_time"] = {
                "active": active,
                "duration_seconds": duration_seconds,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "last_power_above_10w": (
                    last_power_above_10w.isoformat() if last_power_above_10w else None
                ),
                "zero_power_since": zero_power_since.isoformat() if zero_power_since else None,
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.debug(f"Production time updated: active={active}, duration={duration_seconds}s")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to update production time: {e}", exc_info=True)
            return False

    async def update_peak_today(self, power_w: float, timestamp: datetime) -> bool:
        """Update todays peak power LIVE @zara"""
        try:
            data = await self.load_daily_forecasts()

            if "today" not in data:
                data["today"] = self._daily_forecasts_default["today"].copy()

            current_peak = data["today"]["peak_today"].get("power_w", 0.0)
            if power_w > current_peak:
                data["today"]["peak_today"] = {
                    "power_w": round(float(power_w), 2),
                    "at": timestamp.isoformat(),
                }

                await self._atomic_write_json(self.daily_forecasts_file, data)

                _LOGGER.info(f"New peak today: {power_w:.2f}W at {timestamp.strftime('%H:%M:%S')}")

                all_time_peak = (
                    data.get("statistics", {}).get("all_time_peak", {}).get("power_w", 0.0)
                )
                if power_w > all_time_peak:
                    await self.update_all_time_peak(power_w, timestamp)

                return True

            return False

        except Exception as e:
            _LOGGER.error(f"Failed to update peak today: {e}", exc_info=True)
            return False

    async def update_all_time_peak(self, power_w: float, timestamp: datetime) -> bool:
        """Update all-time peak power @zara

        Note: all_time_peak is stored ONLY in daily_forecasts.json (Single Source of Truth).
        The previous synchronization to astronomy_cache.json was removed because:
        1. It was never read - only written (dead code)
        2. It caused race conditions on Raspberry Pi (file locking issues)
        3. astronomy_cache.json uses hourly_max_peaks (kWh) for Clear-Sky, not all_time_peak (kW)
        """
        try:
            data = await self.load_daily_forecasts()

            if "statistics" not in data:
                data["statistics"] = self._daily_forecasts_default["statistics"].copy()

            data["statistics"]["all_time_peak"] = {
                "power_w": round(float(power_w), 2),
                "date": timestamp.date().isoformat(),
                "at": timestamp.isoformat(),
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(f"NEW ALL-TIME PEAK: {power_w:.2f}W on {timestamp.date()}")

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to update all-time peak: {e}", exc_info=True)
            return False

    async def get_all_time_peak(self) -> Optional[float]:
        """Get all-time peak value @zara"""
        try:
            data = await self.load_daily_forecasts()
            return data.get("statistics", {}).get("all_time_peak", {}).get("power_w")
        except Exception:
            return None

    async def finalize_today(
        self,
        yield_kwh: float,
        consumption_kwh: Optional[float] = None,
        production_seconds: int = 0,
        excluded_hours_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Finalize today with actual values at 2330"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()
            today_str = now_local.date().isoformat()

            if "today" not in data or data["today"].get("date") != today_str:
                _LOGGER.error(f"Cannot finalize: today block not set for {today_str}")
                return False

            hours = production_seconds // 3600
            minutes = (production_seconds % 3600) // 60
            production_hours = f"{hours}h {minutes}m"

            accuracy_percent = None
            forecast_kwh = data["today"]["forecast_day"].get("prediction_kwh")
            if forecast_kwh and forecast_kwh > 0:
                error = abs(forecast_kwh - yield_kwh)
                accuracy = max(0.0, 100.0 - (error / forecast_kwh * 100))
                accuracy_percent = round(accuracy, 1)

            finalized_data = {
                "yield_kwh": round(float(yield_kwh), 2),
                "consumption_kwh": (
                    round(float(consumption_kwh), 2) if consumption_kwh is not None else None
                ),
                "production_hours": production_hours,
                "accuracy_percent": accuracy_percent,
                "at": now_local.isoformat(),
            }

            # Add excluded hours info if available
            if excluded_hours_info:
                finalized_data["excluded_hours"] = excluded_hours_info

            data["today"]["finalized"] = finalized_data

            await self._atomic_write_json(self.daily_forecasts_file, data)

            excluded_info_str = ""
            if excluded_hours_info:
                excluded_info_str = f", Excluded={excluded_hours_info.get('count', 0)}/{excluded_hours_info.get('total', 0)} hours"

            _LOGGER.info(
                f"[OK] Today finalized: Yield={yield_kwh:.2f} kWh, "
                f"Consumption={f'{consumption_kwh:.2f}' if consumption_kwh is not None else 'N/A'} kWh, "
                f"Production={production_hours}, "
                f"Accuracy={f'{accuracy_percent:.1f}' if accuracy_percent else 'N/A'}%"
                f"{excluded_info_str}"
            )

            await self._update_aggregated_data(yield_kwh, consumption_kwh)

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to finalize today: {e}", exc_info=True)
            return False

    async def _update_aggregated_data(
        self, yield_kwh: float, consumption_kwh: Optional[float] = None
    ) -> bool:
        """Update current week month and rolling period statistics"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()

            history = data.get("history", [])

            iso = now_local.isocalendar()
            current_week_period = f"{iso[0]}-W{iso[1]:02d}"
            monday = now_local - timedelta(days=now_local.weekday())
            sunday = monday + timedelta(days=6)
            week_start = monday.date().isoformat()
            week_end = sunday.date().isoformat()

            current_week_entries = [
                entry
                for entry in history
                if week_start <= entry.get("date", "") <= week_end
                and entry.get("actual_kwh") is not None
            ]

            if current_week_entries:
                week_yield = sum(e["actual_kwh"] for e in current_week_entries)
                week_consumption = sum(
                    e.get("consumption_kwh", 0)
                    for e in current_week_entries
                    if e.get("consumption_kwh")
                )

                data["statistics"]["current_week"] = {
                    "period": current_week_period,
                    "date_range": f"{week_start}/{week_end}",
                    "yield_kwh": round(week_yield, 2),
                    "consumption_kwh": round(week_consumption, 2),
                    "days": len(current_week_entries),
                    "updated_at": now_local.isoformat(),
                }

            current_month_period = now_local.strftime("%Y-%m")
            month_start = now_local.replace(day=1).date().isoformat()

            current_month_entries = [
                entry
                for entry in history
                if entry.get("date", "").startswith(current_month_period)
                and entry.get("actual_kwh") is not None
            ]

            if current_month_entries:
                month_yield = sum(e["actual_kwh"] for e in current_month_entries)
                month_consumption = sum(
                    e.get("consumption_kwh", 0)
                    for e in current_month_entries
                    if e.get("consumption_kwh")
                )

                autarky_values = []
                for e in current_month_entries:
                    if e.get("autarky") is not None:
                        autarky_values.append(e["autarky"])
                avg_autarky = sum(autarky_values) / len(autarky_values) if autarky_values else 0.0

                data["statistics"]["current_month"] = {
                    "period": current_month_period,
                    "yield_kwh": round(month_yield, 2),
                    "consumption_kwh": round(month_consumption, 2),
                    "avg_autarky": round(avg_autarky, 1),
                    "days": len(current_month_entries),
                    "updated_at": now_local.isoformat(),
                }

            cutoff_7 = (now_local - timedelta(days=7)).date().isoformat()
            last_7_days = [
                entry
                for entry in history
                if entry.get("date", "") >= cutoff_7 and entry.get("actual_kwh") is not None
            ]

            if last_7_days:
                avg_yield_7 = sum(e["actual_kwh"] for e in last_7_days) / len(last_7_days)
                total_yield_7 = sum(e["actual_kwh"] for e in last_7_days)

                accuracies_7 = [e["accuracy"] for e in last_7_days if e.get("accuracy") is not None]
                avg_accuracy_7 = sum(accuracies_7) / len(accuracies_7) if accuracies_7 else 0.0

                data["statistics"]["last_7_days"] = {
                    "avg_yield_kwh": round(avg_yield_7, 2),
                    "avg_accuracy": round(avg_accuracy_7, 1),
                    "total_yield_kwh": round(total_yield_7, 2),
                    "calculated_at": now_local.isoformat(),
                }

            cutoff_30 = (now_local - timedelta(days=30)).date().isoformat()
            last_30_days = [
                entry
                for entry in history
                if entry.get("date", "") >= cutoff_30 and entry.get("actual_kwh") is not None
            ]

            if last_30_days:
                avg_yield_30 = sum(e["actual_kwh"] for e in last_30_days) / len(last_30_days)
                total_yield_30 = sum(e["actual_kwh"] for e in last_30_days)

                accuracies_30 = [
                    e["accuracy"] for e in last_30_days if e.get("accuracy") is not None
                ]
                avg_accuracy_30 = sum(accuracies_30) / len(accuracies_30) if accuracies_30 else 0.0

                data["statistics"]["last_30_days"] = {
                    "avg_yield_kwh": round(avg_yield_30, 2),
                    "avg_accuracy": round(avg_accuracy_30, 1),
                    "total_yield_kwh": round(total_yield_30, 2),
                    "calculated_at": now_local.isoformat(),
                }

            cutoff_365 = (now_local - timedelta(days=365)).date().isoformat()
            last_365_days = [
                entry
                for entry in history
                if entry.get("date", "") >= cutoff_365 and entry.get("actual_kwh") is not None
            ]

            if last_365_days:
                avg_yield_365 = sum(e["actual_kwh"] for e in last_365_days) / len(last_365_days)
                total_yield_365 = sum(e["actual_kwh"] for e in last_365_days)

                data["statistics"]["last_365_days"] = {
                    "avg_yield_kwh": round(avg_yield_365, 2),
                    "total_yield_kwh": round(total_yield_365, 2),
                    "calculated_at": now_local.isoformat(),
                }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(
                f"[OK] Statistics calculated: "
                f"7d avg={data['statistics']['last_7_days'].get('avg_yield_kwh', 0):.2f} kWh, "
                f"30d avg={data['statistics']['last_30_days'].get('avg_yield_kwh', 0):.2f} kWh"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to calculate statistics: {e}", exc_info=True)
            return False

    async def get_history(
        self, days: int = 30, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get history entries with optional filtering"""
        try:
            data = await self.load_daily_forecasts()
            history = data.get("history", [])

            if start_date and end_date:
                history = [
                    entry for entry in history if start_date <= entry.get("date", "") <= end_date
                ]
            elif days:
                cutoff = (dt_util.now() - timedelta(days=days)).date().isoformat()
                history = [entry for entry in history if entry.get("date", "") >= cutoff]

            return history

        except Exception as e:
            _LOGGER.error(f"Failed to get history: {e}")
            return []

    async def rotate_forecasts_at_midnight(self) -> bool:
        """Rotate forecasts at midnight 000030 @zara"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()
            new_today_str = now_local.date().isoformat()

            _LOGGER.info(f"Starting midnight forecast rotation for new day: {new_today_str}")

            tomorrow_forecast = data.get("today", {}).get("forecast_tomorrow", {})
            tomorrow_date = tomorrow_forecast.get("date")

            day_after_forecast = data.get("today", {}).get("forecast_day_after_tomorrow", {})

            tomorrow_date_str = (now_local + timedelta(days=1)).date().isoformat()

            data["today"] = {
                "date": new_today_str,
                "forecast_day": {
                    "prediction_kwh": tomorrow_forecast.get("prediction_kwh"),
                    "locked": False,
                    "locked_at": None,
                    "source": tomorrow_forecast.get("source", "rotated_from_tomorrow"),
                    "next_update": None,
                    "updates": [],
                },
                "forecast_tomorrow": {
                    "date": tomorrow_date_str,
                    "prediction_kwh": day_after_forecast.get("prediction_kwh"),
                    "locked": False,
                    "locked_at": None,
                    "source": day_after_forecast.get("source", "rotated_from_day_after"),
                    "next_update": None,
                    "updates": day_after_forecast.get("updates", [])[-10:],
                },
                "forecast_day_after_tomorrow": {
                    "date": None,
                    "prediction_kwh": None,
                    "locked": False,
                    "locked_at": None,
                    "source": None,
                    "next_update": None,
                    "updates": [],
                },
                "forecast_best_hour": {
                    "hour": None,
                    "prediction_kwh": None,
                    "locked": False,
                    "locked_at": None,
                    "source": None,
                },
                "actual_best_hour": {"hour": None, "actual_kwh": None, "saved_at": None},
                "forecast_next_hour": {
                    "period": None,
                    "prediction_kwh": None,
                    "updated_at": None,
                    "source": None,
                },
                "production_time": {
                    "active": False,
                    "duration_seconds": 0,
                    "start_time": None,
                    "end_time": None,
                    "last_power_above_10w": None,
                    "zero_power_since": None,
                },
                "peak_today": {"power_w": 0.0, "at": None},
                "yield_today": {"kwh": None, "sensor": None},
                "consumption_today": {"kwh": None, "sensor": None},
                "autarky": {"percent": None, "calculated_at": None},
                "finalized": None,
            }

            tomorrow_str = (now_local + timedelta(days=1)).date().isoformat()
            data["tomorrow"] = {
                "date": tomorrow_str,
                "forecast_day": {
                    "prediction_kwh": day_after_forecast.get("prediction_kwh"),
                    "locked": False,
                    "locked_at": None,
                    "source": day_after_forecast.get("source", "rotated_from_day_after"),
                    "next_update": None,
                    "updates": [],
                },
            }

            day_after_str = (now_local + timedelta(days=2)).date().isoformat()
            data["day_after_tomorrow"] = {
                "date": day_after_str,
                "prediction_kwh": None,
                "locked": False,
                "locked_at": None,
                "source": None,
                "next_update": None,
                "updates": [],
            }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(
                f"[OK] Midnight rotation complete: "
                f"today={new_today_str} (unlocked, {data['today']['forecast_day'].get('prediction_kwh') or 0:.2f} kWh), "
                f"today.tomorrow={tomorrow_date_str} (unlocked, {data['today']['forecast_tomorrow'].get('prediction_kwh') or 0:.2f} kWh from rotated day_after), "
                f"today.day_after=(empty)"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to rotate forecasts at midnight: {e}", exc_info=True)
            return False

    async def move_to_history(self) -> bool:
        """Move finalized today data to history runs at 2331 @zara"""
        try:
            data = await self.load_daily_forecasts()
            today_data = data.get("today", {})

            finalized_data = today_data.get("finalized")
            if not finalized_data:
                _LOGGER.warning("Cannot move to history: Today has not been finalized yet")
                return False

            date_str = today_data.get("date")
            if not date_str:
                _LOGGER.error("Cannot move to history: Today's date is missing")
                return False

            forecast_day = today_data.get("forecast_day", {})
            predicted_kwh = forecast_day.get("prediction_kwh", 0.0)
            forecast_source = forecast_day.get("source", "unknown")

            peak_today = today_data.get("peak_today", {})
            peak_power_w = peak_today.get("power_w")
            peak_at_full = peak_today.get("at")

            peak_at = None
            if peak_at_full:
                try:
                    from datetime import datetime

                    peak_dt = datetime.fromisoformat(peak_at_full)
                    peak_at = peak_dt.strftime("%H:%M:%S")
                except Exception as e:
                    _LOGGER.debug(f"Could not parse peak timestamp '{peak_at_full}': {e}")
                    peak_at = peak_at_full

            autarky_data = today_data.get("autarky", {})
            autarky_percent = autarky_data.get("percent")

            production_seconds = 0
            production_hours_str = finalized_data.get("production_hours", "0h 0m")
            try:

                parts = production_hours_str.replace("h", "").replace("m", "").split()
                if len(parts) >= 2:
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    production_seconds = hours * 3600 + minutes * 60
            except Exception as e:
                _LOGGER.debug(f"Could not parse production_hours '{production_hours_str}': {e}")

            history_entry = {
                "date": date_str,
                "predicted_kwh": predicted_kwh or 0.0,
                "actual_kwh": finalized_data.get("yield_kwh") or 0.0,
                "accuracy": finalized_data.get("accuracy_percent") or 0.0,
                "consumption_kwh": finalized_data.get("consumption_kwh") or 0.0,
                "autarky": autarky_percent or 0.0,
                "production_hours": production_hours_str,
                "peak_power_w": peak_power_w or 0.0,
                "peak_at": peak_at,
                "forecast_source": forecast_source,
                "finalized_at": finalized_data.get("at"),
            }

            # Add excluded hours info if available
            excluded_hours_info = finalized_data.get("excluded_hours")
            if excluded_hours_info:
                history_entry["excluded_hours"] = excluded_hours_info

            if "history" not in data:
                data["history"] = []

            existing_dates = [entry.get("date") for entry in data["history"]]
            if date_str in existing_dates:
                _LOGGER.warning(
                    f"History already contains entry for {date_str} - "
                    f"removing old entry and replacing with new finalized data"
                )

                data["history"] = [entry for entry in data["history"] if entry.get("date") != date_str]

            data["history"].insert(0, history_entry)

            if len(data["history"]) > 730:
                data["history"] = data["history"][:730]

            data["metadata"]["history_entries"] = len(data["history"])

            await self._atomic_write_json(self.daily_forecasts_file, data)

            forecast_val = history_entry["predicted_kwh"] or 0.0
            actual_val = history_entry["actual_kwh"] or 0.0
            accuracy_val = history_entry["accuracy"] or 0.0

            _LOGGER.info(
                f"Moved to history: {date_str} - "
                f"Forecast={forecast_val:.2f} kWh, "
                f"Actual={actual_val:.2f} kWh, "
                f"Accuracy={accuracy_val:.1f}%"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to move to history: {e}", exc_info=True)
            return False

    async def calculate_statistics(self) -> bool:
        """Calculate aggregated statistics from history runs at 2332 @zara"""
        try:
            data = await self.load_daily_forecasts()
            now_local = dt_util.now()

            history = data.get("history", [])

            if not history:
                _LOGGER.warning("No history data available to calculate statistics")
                return True

            cutoff_7 = (now_local - timedelta(days=7)).date().isoformat()
            last_7_days = [
                entry
                for entry in history
                if entry.get("date", "") >= cutoff_7 and entry.get("actual_kwh") is not None
            ]

            if last_7_days:
                avg_yield_7 = sum(e["actual_kwh"] for e in last_7_days) / len(last_7_days)
                total_yield_7 = sum(e["actual_kwh"] for e in last_7_days)

                accuracies_7 = [e["accuracy"] for e in last_7_days if e.get("accuracy") is not None]
                avg_accuracy_7 = sum(accuracies_7) / len(accuracies_7) if accuracies_7 else 0.0

                data["statistics"]["last_7_days"] = {
                    "avg_yield_kwh": round(avg_yield_7, 2),
                    "avg_accuracy": round(avg_accuracy_7, 1),
                    "total_yield_kwh": round(total_yield_7, 2),
                    "calculated_at": now_local.isoformat(),
                }

            cutoff_30 = (now_local - timedelta(days=30)).date().isoformat()
            last_30_days = [
                entry
                for entry in history
                if entry.get("date", "") >= cutoff_30 and entry.get("actual_kwh") is not None
            ]

            if last_30_days:
                avg_yield_30 = sum(e["actual_kwh"] for e in last_30_days) / len(last_30_days)
                total_yield_30 = sum(e["actual_kwh"] for e in last_30_days)

                accuracies_30 = [
                    e["accuracy"] for e in last_30_days if e.get("accuracy") is not None
                ]
                avg_accuracy_30 = sum(accuracies_30) / len(accuracies_30) if accuracies_30 else 0.0

                data["statistics"]["last_30_days"] = {
                    "avg_yield_kwh": round(avg_yield_30, 2),
                    "avg_accuracy": round(avg_accuracy_30, 1),
                    "total_yield_kwh": round(total_yield_30, 2),
                    "calculated_at": now_local.isoformat(),
                }

            cutoff_365 = (now_local - timedelta(days=365)).date().isoformat()
            last_365_days = [
                entry
                for entry in history
                if entry.get("date", "") >= cutoff_365 and entry.get("actual_kwh") is not None
            ]

            if last_365_days:
                avg_yield_365 = sum(e["actual_kwh"] for e in last_365_days) / len(last_365_days)
                total_yield_365 = sum(e["actual_kwh"] for e in last_365_days)

                data["statistics"]["last_365_days"] = {
                    "avg_yield_kwh": round(avg_yield_365, 2),
                    "total_yield_kwh": round(total_yield_365, 2),
                    "calculated_at": now_local.isoformat(),
                }

            iso = now_local.isocalendar()
            current_week_period = f"{iso[0]}-W{iso[1]:02d}"
            monday = now_local - timedelta(days=now_local.weekday())
            sunday = monday + timedelta(days=6)
            week_start = monday.date().isoformat()
            week_end = sunday.date().isoformat()

            current_week_entries = [
                entry
                for entry in history
                if week_start <= entry.get("date", "") <= week_end
                and entry.get("actual_kwh") is not None
            ]

            if current_week_entries:
                week_yield = sum(e["actual_kwh"] for e in current_week_entries)
                week_consumption = sum(
                    e.get("consumption_kwh", 0)
                    for e in current_week_entries
                    if e.get("consumption_kwh")
                )

                data["statistics"]["current_week"] = {
                    "period": current_week_period,
                    "date_range": f"{week_start}/{week_end}",
                    "yield_kwh": round(week_yield, 2),
                    "consumption_kwh": round(week_consumption, 2),
                    "days": len(current_week_entries),
                    "updated_at": now_local.isoformat(),
                }

            current_month_period = now_local.strftime("%Y-%m")

            current_month_entries = [
                entry
                for entry in history
                if entry.get("date", "").startswith(current_month_period)
                and entry.get("actual_kwh") is not None
            ]

            if current_month_entries:
                month_yield = sum(e["actual_kwh"] for e in current_month_entries)
                month_consumption = sum(
                    e.get("consumption_kwh", 0)
                    for e in current_month_entries
                    if e.get("consumption_kwh")
                )

                autarky_values = []
                for e in current_month_entries:
                    if e.get("autarky") is not None:
                        autarky_values.append(e["autarky"])
                avg_autarky = sum(autarky_values) / len(autarky_values) if autarky_values else 0.0

                data["statistics"]["current_month"] = {
                    "period": current_month_period,
                    "yield_kwh": round(month_yield, 2),
                    "consumption_kwh": round(month_consumption, 2),
                    "avg_autarky": round(avg_autarky, 1),
                    "days": len(current_month_entries),
                    "updated_at": now_local.isoformat(),
                }

            await self._atomic_write_json(self.daily_forecasts_file, data)

            _LOGGER.info(
                f"Statistics calculated: "
                f"7d avg={data['statistics'].get('last_7_days', {}).get('avg_yield_kwh', 0):.2f} kWh, "
                f"30d avg={data['statistics'].get('last_30_days', {}).get('avg_yield_kwh', 0):.2f} kWh"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to calculate statistics: {e}", exc_info=True)
            return False

    async def save_multi_day_hourly_forecast(
        self,
        hourly_forecast: List[Dict[str, Any]],
    ) -> bool:
        """Save hourly forecasts for today, tomorrow, and day after tomorrow.

        Creates a rolling JSON file with hourly breakdown per day.
        File: stats/multi_day_hourly_forecast.json

        @zara
        """
        try:
            if not hourly_forecast:
                return False

            now_local = dt_util.now()
            today_str = now_local.date().isoformat()
            tomorrow_str = (now_local + timedelta(days=1)).date().isoformat()
            day_after_str = (now_local + timedelta(days=2)).date().isoformat()

            forecast_file = self.data_dir / "stats" / "multi_day_hourly_forecast.json"

            by_date: Dict[str, List[Dict[str, Any]]] = {
                today_str: [],
                tomorrow_str: [],
                day_after_str: [],
            }

            for entry in hourly_forecast:
                # Support multiple date field names from different sources
                date_str = entry.get("date")
                if not date_str:
                    # Fallback to datetime or local_datetime field
                    dt_field = entry.get("datetime") or entry.get("local_datetime")
                    if dt_field:
                        if isinstance(dt_field, str):
                            date_str = dt_field[:10]
                        elif hasattr(dt_field, "date"):
                            date_str = dt_field.date().isoformat()

                if not date_str or date_str not in by_date:
                    continue

                hour = entry.get("hour")
                if hour is None:
                    dt_field = entry.get("datetime") or entry.get("local_datetime")
                    if isinstance(dt_field, str) and len(dt_field) >= 13:
                        hour = int(dt_field[11:13])

                # Support multiple field names for production value
                production = entry.get("production_kwh") or entry.get("solar_kwh", 0)

                hourly_entry = {
                    "hour": hour,
                    "prediction_kwh": round(production, 4),
                    "cloud_cover": entry.get("cloud_cover"),
                    "temperature": entry.get("temperature"),
                    "solar_radiation_wm2": entry.get("solar_radiation_wm2"),
                    "weather_source": entry.get("weather_source"),
                }

                # Support both old "groups" format and new "panel_group_predictions" format
                if "groups" in entry:
                    hourly_entry["panel_groups"] = entry["groups"]
                    # Convert to AI-compatible format for TinyLSTM learning
                    if "panel_group_predictions" not in entry:
                        hourly_entry["panel_group_predictions"] = {
                            g["name"]: round(g.get("power_kwh", 0), 4)
                            for g in entry["groups"]
                        }
                if "panel_groups" in entry:
                    hourly_entry["panel_groups"] = entry["panel_groups"]
                if "panel_group_predictions" in entry:
                    hourly_entry["panel_group_predictions"] = entry["panel_group_predictions"]

                by_date[date_str].append(hourly_entry)

            for date_str in by_date:
                by_date[date_str].sort(key=lambda x: x.get("hour", 0))

            totals = {}
            for date_str, hours in by_date.items():
                totals[date_str] = round(
                    sum(h.get("prediction_kwh", 0) for h in hours), 3
                )

            data = {
                "version": "1.0",
                "updated_at": now_local.isoformat(),
                "days": {
                    today_str: {
                        "date": today_str,
                        "day_type": "today",
                        "total_kwh": totals.get(today_str, 0),
                        "hourly": by_date.get(today_str, []),
                    },
                    tomorrow_str: {
                        "date": tomorrow_str,
                        "day_type": "tomorrow",
                        "total_kwh": totals.get(tomorrow_str, 0),
                        "hourly": by_date.get(tomorrow_str, []),
                    },
                    day_after_str: {
                        "date": day_after_str,
                        "day_type": "day_after_tomorrow",
                        "total_kwh": totals.get(day_after_str, 0),
                        "hourly": by_date.get(day_after_str, []),
                    },
                },
                "summary": {
                    "today": totals.get(today_str, 0),
                    "tomorrow": totals.get(tomorrow_str, 0),
                    "day_after_tomorrow": totals.get(day_after_str, 0),
                },
            }

            await self._atomic_write_json(forecast_file, data)

            _LOGGER.debug(
                f"Multi-day hourly forecast saved: today={totals.get(today_str, 0):.2f}, "
                f"tomorrow={totals.get(tomorrow_str, 0):.2f}, "
                f"day_after={totals.get(day_after_str, 0):.2f} kWh"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Failed to save multi-day hourly forecast: {e}", exc_info=True)
            return False

    async def load_multi_day_hourly_forecast(self) -> Optional[Dict[str, Any]]:
        """Load the multi-day hourly forecast from file.

        @zara
        """
        try:
            forecast_file = self.data_dir / "stats" / "multi_day_hourly_forecast.json"

            if not forecast_file.exists():
                return None

            return await self._read_json_file(forecast_file, default={})

        except Exception as e:
            _LOGGER.error(f"Failed to load multi-day hourly forecast: {e}", exc_info=True)
            return None
