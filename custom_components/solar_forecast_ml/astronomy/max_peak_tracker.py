# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""
Max Peak Tracker for Solar Forecast ML V16.2.0.
Tracks and updates maximum PV output records per hour using database storage.
"""

import logging
from datetime import date
from typing import Dict, Optional

from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)


class MaxPeakTracker:
    """Track and update maximum PV output records per hour in database. @zara"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def check_and_update_peak(
        self,
        target_date: date,
        target_hour: int,
        actual_kwh: float,
        conditions: Optional[Dict] = None,
    ) -> bool:
        """Check if this is a new peak for the hour and update database if so. @zara"""
        try:
            current_peak = await self.db.fetchone(
                "SELECT kwh FROM astronomy_hourly_peaks WHERE hour = ?",
                (target_hour,)
            )

            current_max = current_peak[0] if current_peak else 0.0

            if actual_kwh <= current_max:
                return False

            conditions = conditions or {}
            await self.db.execute(
                """INSERT INTO astronomy_hourly_peaks
                   (hour, kwh, date, sun_elevation_deg, cloud_cover_percent,
                    temperature_c, solar_radiation_wm2)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(hour) DO UPDATE SET
                       kwh = excluded.kwh,
                       date = excluded.date,
                       sun_elevation_deg = excluded.sun_elevation_deg,
                       cloud_cover_percent = excluded.cloud_cover_percent,
                       temperature_c = excluded.temperature_c,
                       solar_radiation_wm2 = excluded.solar_radiation_wm2""",
                (target_hour, actual_kwh, target_date,
                 conditions.get("sun_elevation_deg"),
                 conditions.get("cloud_cover_percent"),
                 conditions.get("temperature_c"),
                 conditions.get("solar_radiation_wm2"))
            )

            global_max = await self.db.fetchone(
                "SELECT max_peak_record_kwh FROM astronomy_system_info WHERE id = 1"
            )
            global_max_kwh = global_max[0] if global_max else 0.0

            if actual_kwh > global_max_kwh:
                await self.db.execute(
                    """UPDATE astronomy_system_info
                       SET max_peak_record_kwh = ?,
                           max_peak_date = ?,
                           max_peak_hour = ?,
                           max_peak_sun_elevation_deg = ?,
                           max_peak_cloud_cover_percent = ?,
                           max_peak_temperature_c = ?,
                           max_peak_solar_radiation_wm2 = ?,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE id = 1""",
                    (actual_kwh, target_date, target_hour,
                     conditions.get("sun_elevation_deg"),
                     conditions.get("cloud_cover_percent"),
                     conditions.get("temperature_c"),
                     conditions.get("solar_radiation_wm2"))
                )

            _LOGGER.info(
                f"New peak record for hour {target_hour}: {actual_kwh:.3f} kWh "
                f"(previous: {current_max:.3f} kWh)"
            )

            return True

        except Exception as e:
            _LOGGER.error(f"Error checking/updating peak: {e}", exc_info=True)
            return False

    async def get_historical_max_for_hour(self, hour: int) -> Optional[float]:
        """Get historical maximum kWh for a specific hour from database. @zara"""
        try:
            row = await self.db.fetchone(
                "SELECT kwh FROM astronomy_hourly_peaks WHERE hour = ?",
                (hour,)
            )

            if row:
                return row[0]
            return 0.0

        except Exception as e:
            _LOGGER.error(f"Error getting historical max: {e}")
            return None

    async def get_all_hourly_peaks(self) -> Dict[int, Dict]:
        """Get all hourly peak records from database. @zara"""
        try:
            rows = await self.db.fetchall(
                """SELECT hour, kwh, date, sun_elevation_deg, cloud_cover_percent,
                          temperature_c, solar_radiation_wm2
                   FROM astronomy_hourly_peaks
                   ORDER BY hour"""
            )

            peaks = {}
            for row in rows:
                peaks[row[0]] = {
                    "kwh": row[1],
                    "date": str(row[2]) if row[2] else None,
                    "conditions": {
                        "sun_elevation_deg": row[3],
                        "cloud_cover_percent": row[4],
                        "temperature_c": row[5],
                        "solar_radiation_wm2": row[6],
                    }
                }

            return peaks

        except Exception as e:
            _LOGGER.error(f"Error getting all hourly peaks: {e}")
            return {}

    async def get_global_max(self) -> Optional[Dict]:
        """Get global maximum peak record from database. @zara"""
        try:
            row = await self.db.fetchone(
                """SELECT max_peak_record_kwh, max_peak_date, max_peak_hour,
                          max_peak_sun_elevation_deg, max_peak_cloud_cover_percent,
                          max_peak_temperature_c, max_peak_solar_radiation_wm2
                   FROM astronomy_system_info WHERE id = 1"""
            )

            if not row or not row[0]:
                return None

            return {
                "kwh": row[0],
                "date": str(row[1]) if row[1] else None,
                "hour": row[2],
                "conditions": {
                    "sun_elevation_deg": row[3],
                    "cloud_cover_percent": row[4],
                    "temperature_c": row[5],
                    "solar_radiation_wm2": row[6],
                }
            }

        except Exception as e:
            _LOGGER.error(f"Error getting global max: {e}")
            return None
