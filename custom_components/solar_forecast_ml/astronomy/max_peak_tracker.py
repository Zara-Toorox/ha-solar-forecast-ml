# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

_LOGGER = logging.getLogger(__name__)

class MaxPeakTracker:
    """Track and update maximum PV output records per hour"""

    def __init__(self, astronomy_cache):
        self.astronomy_cache = astronomy_cache
        self.cache_file = astronomy_cache.cache_file

    async def extract_max_peaks_from_history(self, hourly_predictions_file: Path) -> Dict:
        """Extract max peak records from hourly_predictions.json history @zara"""
        _LOGGER.info("Extracting max peak records from history...")

        def _extract_sync():
            try:
                if not hourly_predictions_file.exists():
                    _LOGGER.warning("hourly_predictions.json not found")
                    return None

                with open(hourly_predictions_file, "r") as f:
                    data = json.load(f)

                predictions = data.get("predictions", [])

                hourly_max_peaks = {}
                for hour in range(24):
                    hourly_max_peaks[str(hour)] = {"kwh": 0.0, "date": None, "conditions": {}}

                global_max = {"kwh": 0.0, "date": None, "hour": None, "conditions": {}}

                processed = 0
                updated_hours = set()

                for pred in predictions:
                    actual_kwh = pred.get("actual_kwh")
                    if actual_kwh is None or actual_kwh == 0.0:
                        continue

                    target_date = pred.get("target_date")
                    target_hour = pred.get("target_hour")

                    if target_date is None or target_hour is None:
                        continue

                    processed += 1
                    hour_str = str(target_hour)

                    if actual_kwh > hourly_max_peaks[hour_str]["kwh"]:

                        weather = pred.get("weather_forecast", {})
                        astro = pred.get("astronomy", {})

                        hourly_max_peaks[hour_str] = {
                            "kwh": round(actual_kwh, 4),
                            "date": target_date,
                            "conditions": {
                                "sun_elevation_deg": astro.get("sun_elevation_deg"),
                                "cloud_cover_percent": weather.get("cloud_cover_percent"),
                                "temperature_c": weather.get("temperature_c"),
                                "solar_radiation_wm2": weather.get("solar_radiation_wm2"),
                            },
                        }
                        updated_hours.add(target_hour)

                    if actual_kwh > global_max["kwh"]:
                        global_max = {
                            "kwh": round(actual_kwh, 4),
                            "date": target_date,
                            "hour": target_hour,
                            "conditions": {
                                "sun_elevation_deg": astro.get("sun_elevation_deg"),
                                "cloud_cover_percent": weather.get("cloud_cover_percent"),
                                "temperature_c": weather.get("temperature_c"),
                                "solar_radiation_wm2": weather.get("solar_radiation_wm2"),
                            },
                        }

                return {
                    "hourly_max_peaks": hourly_max_peaks,
                    "global_max": global_max,
                    "processed_samples": processed,
                    "updated_hours": len(updated_hours),
                }

            except Exception as e:
                _LOGGER.error(f"Error extracting max peaks: {e}", exc_info=True)
                return None

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _extract_sync)

        if not result:
            return {"error": "Failed to extract max peaks"}

        await self._update_cache_with_max_peaks(result["hourly_max_peaks"], result["global_max"])

        _LOGGER.info(
            f"Max peaks extracted: {result['processed_samples']} samples processed, "
            f"{result['updated_hours']} hours updated, "
            f"global max: {result['global_max']['kwh']} kWh at hour {result['global_max']['hour']}"
        )

        return result

    async def _update_cache_with_max_peaks(self, hourly_max_peaks: Dict, global_max: Dict):
        """Update astronomy cache file with max peak data @zara"""

        def _update_sync():
            try:
                if not self.cache_file.exists():
                    _LOGGER.warning("Cache file not found, cannot update max peaks")
                    return False

                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)

                if "pv_system" not in cache:
                    cache["pv_system"] = {}

                cache["pv_system"]["max_peak_record_kwh"] = global_max["kwh"]
                cache["pv_system"]["max_peak_date"] = global_max["date"]
                cache["pv_system"]["max_peak_hour"] = global_max["hour"]
                cache["pv_system"]["max_peak_conditions"] = global_max["conditions"]
                cache["pv_system"]["hourly_max_peaks"] = hourly_max_peaks

                cache["last_updated"] = datetime.now().isoformat()

                if "metadata" not in cache:
                    cache["metadata"] = {}

                if "pv_system" not in cache["metadata"]:
                    cache["metadata"]["pv_system"] = {}

                cache["metadata"]["pv_system"]["max_peak_record_kwh"] = global_max["kwh"]
                cache["metadata"]["pv_system"]["max_peak_date"] = global_max["date"]
                cache["metadata"]["pv_system"]["max_peak_hour"] = global_max["hour"]
                cache["metadata"]["pv_system"]["max_peak_conditions"] = global_max["conditions"]
                cache["metadata"]["pv_system"]["hourly_max_peaks"] = hourly_max_peaks

                cache["metadata"]["last_updated"] = datetime.now().isoformat()

                temp_file = self.cache_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, sort_keys=False, ensure_ascii=False)

                temp_file.replace(self.cache_file)
                return True

            except Exception as e:
                _LOGGER.error(f"Error updating cache with max peaks: {e}", exc_info=True)
                return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _update_sync)

    async def check_and_update_peak(
        self,
        target_date: date,
        target_hour: int,
        actual_kwh: float,
        conditions: Optional[Dict] = None,
    ) -> bool:
        """
        Check if this is a new peak for the hour and update if so

        Args:
            target_date: Date of production
            target_hour: Hour (0-23)
            actual_kwh: Actual production in kWh
            conditions: Optional weather/astronomy conditions

        Returns:
            True if this was a new peak
        """

        def _check_and_update_sync():
            try:
                if not self.cache_file.exists():
                    return False

                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)

                hourly_max_peaks = cache.get("pv_system", {}).get("hourly_max_peaks", {})

                hour_str = str(target_hour)

                if hour_str not in hourly_max_peaks:
                    hourly_max_peaks[hour_str] = {"kwh": 0.0, "date": None, "conditions": {}}

                current_max = hourly_max_peaks[hour_str]["kwh"]

                if actual_kwh <= current_max:
                    return False

                hourly_max_peaks[hour_str] = {
                    "kwh": round(actual_kwh, 4),
                    "date": target_date.isoformat(),
                    "conditions": conditions or {},
                }

                global_max_kwh = cache.get("pv_system", {}).get("max_peak_record_kwh", 0.0)

                if "pv_system" not in cache:
                    cache["pv_system"] = {}

                cache["pv_system"]["hourly_max_peaks"] = hourly_max_peaks

                if actual_kwh > global_max_kwh:
                    cache["pv_system"]["max_peak_record_kwh"] = round(actual_kwh, 4)
                    cache["pv_system"]["max_peak_date"] = target_date.isoformat()
                    cache["pv_system"]["max_peak_hour"] = target_hour
                    cache["pv_system"]["max_peak_conditions"] = conditions or {}

                cache["last_updated"] = datetime.now().isoformat()

                if "metadata" not in cache:
                    cache["metadata"] = {}
                if "pv_system" not in cache["metadata"]:
                    cache["metadata"]["pv_system"] = {}

                cache["metadata"]["pv_system"]["hourly_max_peaks"] = hourly_max_peaks

                if actual_kwh > global_max_kwh:
                    cache["metadata"]["pv_system"]["max_peak_record_kwh"] = round(actual_kwh, 4)
                    cache["metadata"]["pv_system"]["max_peak_date"] = target_date.isoformat()
                    cache["metadata"]["pv_system"]["max_peak_hour"] = target_hour
                    cache["metadata"]["pv_system"]["max_peak_conditions"] = conditions or {}

                cache["metadata"]["last_updated"] = datetime.now().isoformat()

                temp_file = self.cache_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, sort_keys=False, ensure_ascii=False)

                temp_file.replace(self.cache_file)

                _LOGGER.info(
                    f"New peak record for hour {target_hour}: {actual_kwh:.3f} kWh "
                    f"(previous: {current_max:.3f} kWh)"
                )

                return True

            except Exception as e:
                _LOGGER.error(f"Error checking/updating peak: {e}", exc_info=True)
                return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _check_and_update_sync)

    async def get_historical_max_for_hour(self, hour: int) -> Optional[float]:
        """Get historical maximum kWh for a specific hour @zara"""

        def _get_sync():
            try:
                if not self.cache_file.exists():
                    return None

                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cache = json.load(f)

                hourly_max_peaks = cache.get("pv_system", {}).get("hourly_max_peaks", {})
                hour_str = str(hour)

                if hour_str in hourly_max_peaks:
                    return hourly_max_peaks[hour_str].get("kwh", 0.0)

                return 0.0

            except Exception as e:
                _LOGGER.error(f"Error getting historical max: {e}")
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _get_sync)
