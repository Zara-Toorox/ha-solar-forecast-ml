# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

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

    def __init__(self, data_dir: Path):
        """Initialize the report generator. @zara"""
        self.data_dir = data_dir
        self.docs_dir = data_dir / "docs"
        self.report_file = self.docs_dir / "system_report.md"

        # Data file paths
        self.seasonal_file = data_dir / "ai" / "seasonal.json"
        self.forecasts_file = data_dir / "stats" / "daily_forecasts.json"
        self.weights_file = data_dir / "ai" / "learned_weights.json"

    async def generate_report(self) -> bool:
        """Generate the system report. @zara"""
        try:
            _LOGGER.info("Generating monthly system report...")

            # Load all data sources
            seasonal_data = await self._load_json(self.seasonal_file)
            forecasts_data = await self._load_json(self.forecasts_file)
            weights_data = await self._load_json(self.weights_file)

            # Build report content
            content = self._build_report(seasonal_data, forecasts_data, weights_data)

            # Write report
            await self._write_report(content)

            _LOGGER.info(f"System report generated: {self.report_file}")
            return True

        except Exception as e:
            _LOGGER.error(f"Failed to generate system report: {e}", exc_info=True)
            return False

    async def _load_json(self, file_path: Path) -> dict:
        """Load JSON file asynchronously. @zara"""
        def _read_sync() -> dict:
            if not file_path.exists():
                return {}
            with open(file_path, "r") as f:
                return json.load(f)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _read_sync)

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
        geometry_data: dict,
        forecasts_data: dict,
        weights_data: dict,
    ) -> str:
        """Build the Markdown report content. @zara"""
        now = datetime.now()

        # Extract data
        estimate = geometry_data.get("estimate", {})
        metadata = geometry_data.get("metadata", {})
        data_points = geometry_data.get("data_points", [])
        statistics = forecasts_data.get("statistics", {})
        history = forecasts_data.get("history", [])

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

        # Split system detection
        is_split_system = self._detect_split_system(data_points)

        # Record peak
        all_time_peak = statistics.get("all_time_peak", {})
        peak_power_kw = all_time_peak.get("power_w", 0) / 1000
        peak_date = all_time_peak.get("date", "N/A")

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
            "## 🔧 Panel Geometry (learned)",
            "",
            "| Parameter | Learned | Configured | Δ |",
            "|-----------|---------|------------|---|",
            f"| Tilt | {learned_tilt:.1f}° | {configured_tilt:.1f}° | {learned_tilt - configured_tilt:+.1f}° |",
            f"| Azimuth | {learned_azimuth:.1f}° | {configured_azimuth:.1f}° | {learned_azimuth - configured_azimuth:+.1f}° |",
            "",
            f"**Orientation:** {orientation}",
            "",
            f"**Split System:** {'Yes' if is_split_system else 'No'}",
            "",
            f"**Confidence:** {confidence:.0f}% · **Samples:** {samples} · **RMSE:** {rmse:.3f} kWh",
            "",
            "---",
            "",
            "## ⚡ Performance",
            "",
            f"**Record Peak:** {peak_power_kw:.2f} kW ({peak_date})",
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

        lines.extend([
            "",
            "---",
            "",
            "## 🖖 Message from the Captain's Log",
            "",
            f"> *\"{star_trek_quote}\"*",
            "",
            "Live long and prosper! 🖖",
            "",
            "---",
            "",
            "*Report by Solar Forecast ML v10.2.0*",
            "",
            "*Created with ☀️ by [Zara-Toorox](https://github.com/Zara-Toorox)*",
        ])

        return "\n".join(lines)

    def _get_star_trek_quote(self) -> str:
        """Return a random Star Trek quote related to energy/sun/exploration. @zara"""
        import random

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

    def _detect_split_system(self, data_points: list) -> bool:
        """Detect if system has split orientation (e.g., East/West). @zara

        A split system typically has panels facing two different directions,
        resulting in two distinct azimuth clusters (e.g., ~90° and ~270°).
        """
        if len(data_points) < 10:
            return False

        azimuths = [p.get("sun_azimuth_deg", 180) for p in data_points]

        # Check for bimodal distribution
        # Split systems typically have morning (East, ~90-150°) and afternoon (West, ~210-270°) peaks
        morning_count = sum(1 for a in azimuths if 90 <= a <= 150)
        afternoon_count = sum(1 for a in azimuths if 210 <= a <= 270)

        total = len(azimuths)

        # If significant production in both morning AND afternoon ranges
        # with similar proportions, it's likely a split system
        if morning_count > 0 and afternoon_count > 0:
            morning_ratio = morning_count / total
            afternoon_ratio = afternoon_count / total

            # Both should have at least 20% of samples
            if morning_ratio > 0.2 and afternoon_ratio > 0.2:
                # And similar proportions (within 2:1 ratio)
                ratio = morning_ratio / afternoon_ratio if afternoon_ratio > 0 else 0
                if 0.5 <= ratio <= 2.0:
                    return True

        return False

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

    def _calculate_seasonal_stats(self, history: list) -> dict:
        """Calculate production statistics per season. @zara"""
        stats = {
            "winter": {"days": 0, "total": 0, "best_day": 0, "values": []},
            "spring": {"days": 0, "total": 0, "best_day": 0, "values": []},
            "summer": {"days": 0, "total": 0, "best_day": 0, "values": []},
            "autumn": {"days": 0, "total": 0, "best_day": 0, "values": []},
        }

        for entry in history:
            date_str = entry.get("date", "")

            # Get yield - check both formats
            yield_kwh = entry.get("yield_kwh") or entry.get("actual_kwh") or 0

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
                stats[season]["avg_daily"] = (
                    stats[season]["total"] / stats[season]["days"]
                )
            else:
                stats[season]["avg_daily"] = 0

        return stats
