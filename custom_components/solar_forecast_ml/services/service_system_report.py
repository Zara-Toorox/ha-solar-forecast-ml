# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast ML DB-Version
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

# *****************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# Refactored: JSON replaced with DatabaseManager @zara
# *****************************************************************************

"""
System report generator service for Solar Forecast ML.
Generates monthly system reports in Markdown format.
Uses DatabaseManager for all data operations.
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.db_manager import DatabaseManager

_LOGGER = logging.getLogger(__name__)

# Season definitions
SEASONS = {
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
}


class SystemReportGenerator:
    """Generates monthly system reports in Markdown format. @zara"""

    def __init__(self, data_dir: Path, db_manager: Optional[DatabaseManager] = None):
        """Initialize the report generator. @zara"""
        self.data_dir = data_dir
        self.docs_dir = data_dir / "docs"
        self.report_file = self.docs_dir / "system_report.md"
        self._db = db_manager

    def set_db_manager(self, db_manager: DatabaseManager) -> None:
        """Set database manager after initialization. @zara"""
        self._db = db_manager

    async def generate_report(self) -> bool:
        """Generate the system report. @zara"""
        try:
            _LOGGER.info("Generating monthly system report...")

            # Load all data from database
            geometry_data = await self._load_geometry_data()
            statistics = await self._load_statistics()
            history = await self._load_history()

            # Build report content
            content = self._build_report(geometry_data, statistics, history)

            # Write report
            await self._write_report(content)

            _LOGGER.info(f"System report generated: {self.report_file}")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to generate system report: {e}", exc_info=True)
            return False

    async def _load_geometry_data(self) -> Dict[str, Any]:
        """Load geometry/calibration data from database. @zara"""
        if not self._db:
            return {}

        try:
            # Get learned geometry from coordinator_state
            row = await self._db.fetchone(
                """SELECT value FROM coordinator_state
                   WHERE key = 'geometry_estimate'"""
            )

            if row and row[0]:
                import json

                return json.loads(row[0])

            return {}

        except Exception as e:
            _LOGGER.error(f"Error loading geometry data: {e}")
            return {}

    async def _load_statistics(self) -> Dict[str, Any]:
        """Load statistics from database. @zara"""
        if not self._db:
            return {}

        try:
            stats = {}

            # Get all-time peak
            row = await self._db.fetchone(
                """SELECT MAX(actual_kwh), target_date, target_hour
                   FROM hourly_predictions
                   WHERE actual_kwh IS NOT NULL"""
            )

            if row and row[0]:
                stats["all_time_peak"] = {
                    "power_kwh": row[0],
                    "date": row[1],
                    "hour": row[2],
                }

            # Get total production
            row = await self._db.fetchone(
                """SELECT SUM(actual_kwh)
                   FROM daily_forecasts
                   WHERE actual_kwh IS NOT NULL"""
            )

            if row and row[0]:
                stats["total_production_kwh"] = row[0]

            # Get average accuracy
            row = await self._db.fetchone(
                """SELECT AVG(accuracy)
                   FROM daily_forecasts
                   WHERE accuracy IS NOT NULL AND accuracy > 0"""
            )

            if row and row[0]:
                stats["avg_accuracy"] = row[0]

            return stats

        except Exception as e:
            _LOGGER.error(f"Error loading statistics: {e}")
            return {}

    async def _load_history(self) -> List[Dict[str, Any]]:
        """Load historical data from database. @zara"""
        if not self._db:
            return []

        try:
            rows = await self._db.fetchall(
                """SELECT date, actual_kwh, forecast_kwh, accuracy
                   FROM daily_forecasts
                   WHERE actual_kwh IS NOT NULL AND actual_kwh > 0
                   ORDER BY date DESC
                   LIMIT 365"""
            )

            return [
                {
                    "date": row[0],
                    "actual_kwh": row[1],
                    "forecast_kwh": row[2],
                    "accuracy": row[3],
                }
                for row in rows
            ]

        except Exception as e:
            _LOGGER.error(f"Error loading history: {e}")
            return []

    async def _write_report(self, content: str) -> None:
        """Write report to file asynchronously. @zara"""

        def _write_sync():
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.report_file, "w", encoding="utf-8") as f:
                f.write(content)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_sync)

    def _build_report(
        self,
        geometry_data: Dict[str, Any],
        statistics: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> str:
        """Build the Markdown report content. @zara"""
        now = datetime.now()

        # Extract data
        estimate = geometry_data.get("estimate", {})
        metadata = geometry_data.get("metadata", {})

        # System info
        capacity_kwp = metadata.get("system_capacity_kwp", 0)

        # Geometry
        learned_tilt = estimate.get("tilt_deg", 30.0)
        learned_azimuth = estimate.get("azimuth_deg", 180.0)
        configured_tilt = 30.0  # Default
        configured_azimuth = 180.0  # Default
        confidence = estimate.get("confidence", 0) * 100
        samples = estimate.get("sample_count", 0)
        rmse = estimate.get("error_metrics", {}).get("rmse_kwh", 0)

        # Record peak
        all_time_peak = statistics.get("all_time_peak", {})
        peak_power_kwh = all_time_peak.get("power_kwh", 0)
        peak_date = all_time_peak.get("date", "N/A")

        # Total production
        total_production = statistics.get("total_production_kwh", 0)
        avg_accuracy = statistics.get("avg_accuracy", 0)

        # Seasonal stats
        seasonal_stats = self._calculate_seasonal_stats(history)

        # Orientation text
        orientation = self._azimuth_to_orientation(learned_azimuth)

        # Build Markdown
        lines = [
            "# Solar Forecast ML - System Report",
            f"> Generated: {now.strftime('%Y-%m-%d %H:%M')} | System: {capacity_kwp} kWp",
            "",
            "---",
            "",
            "## Panel Geometry (learned)",
            "",
            "| Parameter | Learned | Configured | Delta |",
            "|-----------|---------|------------|-------|",
            f"| Tilt | {learned_tilt:.1f} deg | {configured_tilt:.1f} deg | {learned_tilt - configured_tilt:+.1f} deg |",
            f"| Azimuth | {learned_azimuth:.1f} deg | {configured_azimuth:.1f} deg | {learned_azimuth - configured_azimuth:+.1f} deg |",
            "",
            f"**Orientation:** {orientation}",
            "",
            f"**Confidence:** {confidence:.0f}% | **Samples:** {samples} | **RMSE:** {rmse:.3f} kWh",
            "",
            "---",
            "",
            "## Performance",
            "",
            f"**Record Peak:** {peak_power_kwh:.2f} kWh ({peak_date})",
            "",
            f"**Total Production:** {total_production:.1f} kWh",
            "",
            f"**Average Accuracy:** {avg_accuracy:.1f}%",
            "",
            "### Seasonal Production",
            "",
            "| Season | Best Day | Avg Daily | Total |",
            "|--------|----------|-----------|-------|",
        ]

        # Add seasonal rows
        for season in ["Winter", "Spring", "Summer", "Autumn"]:
            stats = seasonal_stats.get(season.lower(), {})
            best = stats.get("best_day", 0)
            avg = stats.get("avg_daily", 0)
            total = stats.get("total", 0)
            days = stats.get("days", 0)

            if days > 0:
                lines.append(
                    f"| {season} | {best:.2f} kWh | {avg:.2f} kWh | {total:.1f} kWh |"
                )
            else:
                lines.append(f"| {season} | - | - | - |")

        # Footer with Star Trek quote and greeting
        star_trek_quote = self._get_star_trek_quote()

        lines.extend(
            [
                "",
                "---",
                "",
                "## Message from the Captain's Log",
                "",
                f"> *\"{star_trek_quote}\"*",
                "",
                "Live long and prosper!",
                "",
                "---",
                "",
                "*Report by Solar Forecast ML*",
                "",
                "*Created with solar power by [Zara-Toorox](https://github.com/Zara-Toorox)*",
            ]
        )

        return "\n".join(lines)

    def _get_star_trek_quote(self) -> str:
        """Return a random Star Trek quote related to energy/sun/exploration. @zara"""
        quotes = [
            "The sun is the source of all life. Even in the 24th century, we still look to the stars. - Captain Picard",
            "Infinite diversity in infinite combinations... including solar panel orientations. - Spock",
            "Make it so! And by 'it', I mean maximum solar efficiency. - Captain Picard",
            "Beam me up some photons, Scotty! - Captain Kirk (probably)",
            "Resistance to renewable energy is futile. - The Borg",
            "Logic dictates that harvesting solar energy is the most efficient course of action. - Spock",
            "I'm giving her all she's got, Captain! The panels are at maximum output! - Scotty",
            "Space: the final frontier. Solar panels: the home frontier. - Captain Kirk",
            "Today is a good day to generate clean energy! - Worf",
            "The needs of the many outweigh the needs of the few... use solar power. - Spock",
            "Engage... maximum solar absorption! - Captain Picard",
            "Fascinating. Your panel efficiency has improved by 2.6 degrees. - Spock",
        ]

        return random.choice(quotes)

    def _azimuth_to_orientation(self, azimuth: float) -> str:
        """Convert azimuth angle to cardinal direction. @zara"""
        # Normalize to 0-360
        azimuth = azimuth % 360

        directions = [
            (0, "North"),
            (45, "North-East"),
            (90, "East"),
            (135, "South-East"),
            (180, "South"),
            (225, "South-West"),
            (270, "West"),
            (315, "North-West"),
            (360, "North"),
        ]

        for angle, name in directions:
            if abs(azimuth - angle) <= 22.5:
                return name

        return "South"  # Default

    def _calculate_seasonal_stats(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate production statistics per season. @zara"""
        stats = {
            "winter": {"days": 0, "total": 0, "best_day": 0, "values": []},
            "spring": {"days": 0, "total": 0, "best_day": 0, "values": []},
            "summer": {"days": 0, "total": 0, "best_day": 0, "values": []},
            "autumn": {"days": 0, "total": 0, "best_day": 0, "values": []},
        }

        for entry in history:
            date_str = entry.get("date", "")
            yield_kwh = entry.get("actual_kwh") or 0

            if not date_str or yield_kwh <= 0:
                continue

            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                month = date.month
            except ValueError:
                continue

            # Determine season
            season = None
            for s, months in SEASONS.items():
                if month in months:
                    season = s
                    break

            if season:
                stats[season]["days"] += 1
                stats[season]["total"] += yield_kwh
                stats[season]["values"].append(yield_kwh)
                if yield_kwh > stats[season]["best_day"]:
                    stats[season]["best_day"] = yield_kwh

        # Calculate averages
        for season in stats:
            if stats[season]["days"] > 0:
                stats[season]["avg_daily"] = stats[season]["total"] / stats[season]["days"]
            else:
                stats[season]["avg_daily"] = 0

        return stats

    async def get_quick_stats(self) -> Dict[str, Any]:
        """Get quick statistics without generating full report. @zara"""
        if not self._db:
            return {}

        try:
            stats = {}

            # Today's production
            from ..core.core_helpers import SafeDateTimeUtil as dt_util

            today = dt_util.now().date().isoformat()

            row = await self._db.fetchone(
                """SELECT SUM(actual_kwh)
                   FROM hourly_predictions
                   WHERE target_date = ? AND actual_kwh IS NOT NULL""",
                (today,),
            )

            stats["today_actual_kwh"] = row[0] if row and row[0] else 0

            # This week's production
            week_start = (
                dt_util.now().date() - dt_util.dt.timedelta(days=7)
            ).isoformat()

            row = await self._db.fetchone(
                """SELECT SUM(actual_kwh)
                   FROM daily_forecasts
                   WHERE date >= ? AND actual_kwh IS NOT NULL""",
                (week_start,),
            )

            stats["week_actual_kwh"] = row[0] if row and row[0] else 0

            # This month's production
            month_start = dt_util.now().date().replace(day=1).isoformat()

            row = await self._db.fetchone(
                """SELECT SUM(actual_kwh)
                   FROM daily_forecasts
                   WHERE date >= ? AND actual_kwh IS NOT NULL""",
                (month_start,),
            )

            stats["month_actual_kwh"] = row[0] if row and row[0] else 0

            # Average daily this month
            row = await self._db.fetchone(
                """SELECT AVG(actual_kwh), COUNT(*)
                   FROM daily_forecasts
                   WHERE date >= ? AND actual_kwh IS NOT NULL AND actual_kwh > 0""",
                (month_start,),
            )

            stats["month_avg_daily_kwh"] = row[0] if row and row[0] else 0
            stats["month_days_with_data"] = row[1] if row and row[1] else 0

            return stats

        except Exception as e:
            _LOGGER.error(f"Error getting quick stats: {e}")
            return {}


async def create_system_report_generator(
    data_dir: Path, db_manager: Optional[DatabaseManager] = None
) -> SystemReportGenerator:
    """Factory function to create SystemReportGenerator. @zara"""
    return SystemReportGenerator(data_dir, db_manager)
