# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Panel group seasonal report for SFML Stats. @zara"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from .base import BaseChart
from .styles import (
    ChartStyles,
    MONTH_NAMES_DE,
    MONTH_NAMES_SHORT_DE,
)
from ..const import (
    CHART_DPI,
    SFML_STATS_REPORTS,
    SOLAR_FORECAST_ML_STATS,
    SOLAR_ASTRONOMY_CACHE,
)

if TYPE_CHECKING:
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from ..storage import DataValidator

_LOGGER = logging.getLogger(__name__)


@dataclass
class PanelGroupData:
    """Data for a panel group. @zara"""
    name: str
    power_kwp: float
    azimuth_deg: float
    tilt_deg: float
    color: str


@dataclass
class MonthlyPanelStats:
    """Monthly statistics for a panel group. @zara"""
    month: int
    total_theoretical_kwh: float
    avg_daily_kwh: float
    peak_hour_kwh: float
    peak_hour: int
    days_count: int
    avg_poa_wm2: float


PANEL_COLORS = [
    "#FFD700",
    "#FF8C00",
    "#FF6347",
    "#FF4500",
]

SEASON_COLORS = {
    "winter": "#1a237e",
    "spring": "#2e7d32",
    "summer": "#ff8f00",
    "autumn": "#bf360c",
}

SEASON_NAMES = {
    "winter": "Winter",
    "spring": "Fr\u00fchling",
    "summer": "Sommer",
    "autumn": "Herbst",
}


def get_season(month: int) -> str:
    """Return the season for a given month. @zara"""
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


class PanelSeasonalReportChart(BaseChart):
    """Panel group seasonal report chart. @zara"""

    def __init__(self, validator: "DataValidator") -> None:
        """Initialize the panel seasonal report chart. @zara"""
        super().__init__(validator, figsize=(18, 24))
        self._astronomy_path = validator.config_path / SOLAR_FORECAST_ML_STATS / SOLAR_ASTRONOMY_CACHE

    @property
    def export_path(self) -> Path:
        """Return the export path for reports. @zara"""
        return self._validator.get_export_path(SFML_STATS_REPORTS)

    def get_filename(self, **kwargs) -> str:
        """Return the filename for the report. @zara"""
        today = date.today()
        return f"panel_seasonal_report_{today.strftime('%Y%m%d')}.png"

    async def generate(self, **kwargs) -> "Figure":
        """Generate the complete panel seasonal report. @zara"""
        _LOGGER.info("Generiere Panel-Gruppen Saisonal-Report")

        panel_groups, monthly_stats = await self._load_and_aggregate_data()

        if not panel_groups or not monthly_stats:
            return await self._run_in_executor(self._create_no_data_figure)

        fig = await self._run_in_executor(
            self._generate_sync, panel_groups, monthly_stats
        )

        self._fig = fig
        return fig

    def _generate_sync(
        self,
        panel_groups: list[PanelGroupData],
        monthly_stats: dict,
    ) -> "Figure":
        """Synchronous chart rendering in executor. @zara"""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from .styles import apply_dark_theme

        apply_dark_theme()

        fig = plt.figure(figsize=self._figsize, facecolor=self.styles.background)

        gs = gridspec.GridSpec(
            6, 2,
            figure=fig,
            height_ratios=[0.6, 1.2, 1.2, 1.0, 1.0, 0.8],
            width_ratios=[1, 1],
            hspace=0.35,
            wspace=0.25,
        )

        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, panel_groups, monthly_stats)

        ax_monthly = fig.add_subplot(gs[1, :])
        self._draw_monthly_comparison(ax_monthly, panel_groups, monthly_stats)

        ax_seasonal = fig.add_subplot(gs[2, :])
        self._draw_seasonal_stacked(ax_seasonal, panel_groups, monthly_stats)

        ax_daily_g1 = fig.add_subplot(gs[3, 0])
        self._draw_daily_profile(ax_daily_g1, panel_groups[0] if len(panel_groups) > 0 else None, monthly_stats, 0)

        ax_daily_g2 = fig.add_subplot(gs[3, 1])
        self._draw_daily_profile(ax_daily_g2, panel_groups[1] if len(panel_groups) > 1 else None, monthly_stats, 1)

        ax_radar = fig.add_subplot(gs[4, 0], projection='polar')
        self._draw_season_radar(ax_radar, panel_groups, monthly_stats)

        ax_heatmap = fig.add_subplot(gs[4, 1])
        self._draw_efficiency_heatmap(ax_heatmap, panel_groups, monthly_stats)

        ax_footer = fig.add_subplot(gs[5, :])
        self._draw_footer(ax_footer, panel_groups)

        return fig

    async def _load_and_aggregate_data(self) -> tuple[list[PanelGroupData], dict]:
        """Load astronomy data and aggregate by month. @zara"""
        if not self._astronomy_path.exists():
            _LOGGER.warning("astronomy_cache.json nicht gefunden: %s", self._astronomy_path)
            return [], {}

        try:
            with open(self._astronomy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            _LOGGER.error("Fehler beim Laden von astronomy_cache.json: %s", e)
            return [], {}

        days = data.get("days", {})
        if not days:
            return [], {}

        panel_groups: list[PanelGroupData] = []
        for day_data in days.values():
            hourly = day_data.get("hourly", {})
            for hour_data in hourly.values():
                groups = hour_data.get("theoretical_max_per_group", [])
                if groups:
                    for i, g in enumerate(groups):
                        if i >= len(panel_groups):
                            panel_groups.append(PanelGroupData(
                                name=g.get("name", f"Gruppe {i+1}"),
                                power_kwp=g.get("power_kwp", 0),
                                azimuth_deg=g.get("azimuth_deg", 0),
                                tilt_deg=g.get("tilt_deg", 0),
                                color=PANEL_COLORS[i % len(PANEL_COLORS)],
                            ))
                    break
            if panel_groups:
                break

        if not panel_groups:
            _LOGGER.warning("Keine Panel-Gruppen in astronomy_cache.json gefunden")
            return [], {}

        monthly_data: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))

        hourly_by_season: dict[str, dict[int, dict[int, list]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )

        for day_str, day_data in days.items():
            try:
                day_date = date.fromisoformat(day_str)
            except ValueError:
                continue

            month = day_date.month
            season = get_season(month)
            hourly = day_data.get("hourly", {})

            for hour_str, hour_data in hourly.items():
                try:
                    hour = int(hour_str)
                except ValueError:
                    continue

                groups = hour_data.get("theoretical_max_per_group", [])
                for i, g in enumerate(groups):
                    theoretical_kwh = g.get("theoretical_kwh", 0) or 0
                    poa_wm2 = g.get("poa_wm2", 0) or 0

                    if theoretical_kwh > 0:
                        monthly_data[month][i].append((theoretical_kwh, poa_wm2, hour))
                        hourly_by_season[season][i][hour].append(theoretical_kwh)

        monthly_stats: dict[int, list[MonthlyPanelStats]] = {}

        for month in range(1, 13):
            monthly_stats[month] = []
            for group_idx in range(len(panel_groups)):
                data_points = monthly_data[month][group_idx]

                if data_points:
                    daily_totals = defaultdict(float)
                    hourly_peaks = defaultdict(float)

                    total_kwh = 0
                    total_poa = 0

                    for kwh, poa, hour in data_points:
                        total_kwh += kwh
                        total_poa += poa
                        hourly_peaks[hour] = max(hourly_peaks[hour], kwh)

                    days_count = max(1, len(set(day_str[:7] for day_str in days.keys()
                                               if day_str.startswith(f"2025-{month:02d}"))))

                    peak_hour = max(hourly_peaks, key=hourly_peaks.get) if hourly_peaks else 12
                    peak_kwh = hourly_peaks.get(peak_hour, 0)

                    stats = MonthlyPanelStats(
                        month=month,
                        total_theoretical_kwh=total_kwh,
                        avg_daily_kwh=total_kwh / days_count if days_count > 0 else 0,
                        peak_hour_kwh=peak_kwh,
                        peak_hour=peak_hour,
                        days_count=days_count,
                        avg_poa_wm2=total_poa / len(data_points) if data_points else 0,
                    )
                else:
                    stats = MonthlyPanelStats(
                        month=month,
                        total_theoretical_kwh=0,
                        avg_daily_kwh=0,
                        peak_hour_kwh=0,
                        peak_hour=12,
                        days_count=0,
                        avg_poa_wm2=0,
                    )

                monthly_stats[month].append(stats)

        monthly_stats["hourly_by_season"] = hourly_by_season

        return panel_groups, monthly_stats

    def _create_no_data_figure(self) -> "Figure":
        """Create a figure when no data is available. @zara"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=self._figsize, facecolor=self.styles.background)
        ax.set_facecolor(self.styles.background)
        ax.text(
            0.5, 0.5,
            "Keine Panel-Gruppen-Daten verf\u00fcgbar\n\n"
            "Bitte stelle sicher, dass astronomy_cache.json\n"
            "Panel-Gruppen-Daten enth\u00e4lt.",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=16,
            color=self.styles.text_muted,
        )
        ax.axis("off")
        self._fig = fig
        return fig

    def _draw_header(self, ax: "Axes", panel_groups: list[PanelGroupData], monthly_stats: dict) -> None:
        """Draw header with title and panel info. @zara"""
        ax.axis("off")
        ax.set_facecolor(self.styles.background)

        ax.text(
            0.5, 0.85,
            "\u2600\ufe0f Panel-Gruppen Saisonal-Report",
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            color=self.styles.text_primary,
            ha="center",
            va="top",
        )

        ax.text(
            0.5, 0.65,
            "Theoretische Leistung nach Ausrichtung \u00fcber die Jahreszeiten",
            transform=ax.transAxes,
            fontsize=14,
            color=self.styles.text_secondary,
            ha="center",
            va="top",
        )

        box_props = dict(
            boxstyle="round,pad=0.4",
            facecolor=self.styles.background_card,
            edgecolor=self.styles.border,
            alpha=0.9,
        )

        positions = np.linspace(0.15, 0.85, len(panel_groups))

        for group, x_pos in zip(panel_groups, positions):
            yearly_total = sum(
                monthly_stats[m][panel_groups.index(group)].total_theoretical_kwh
                for m in range(1, 13)
                if m in monthly_stats
            )

            direction = self._azimuth_to_direction(group.azimuth_deg)

            info_text = (
                f"{group.name}\n"
                f"{group.power_kwp:.2f} kWp\n"
                f"{direction} ({group.azimuth_deg:.0f}\u00b0)\n"
                f"Neigung: {group.tilt_deg:.0f}\u00b0\n"
                f"~{yearly_total:.0f} kWh/Jahr"
            )

            ax.text(
                x_pos, 0.25,
                info_text,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                color=group.color,
                ha="center",
                va="center",
                bbox={**box_props, "edgecolor": group.color},
                linespacing=1.4,
            )

    def _azimuth_to_direction(self, azimuth: float) -> str:
        """Convert azimuth to compass direction. @zara"""
        directions = ["N", "NNO", "NO", "ONO", "O", "OSO", "SO", "SSO",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        index = int((azimuth + 11.25) / 22.5) % 16
        return directions[index]

    def _draw_monthly_comparison(self, ax: "Axes", panel_groups: list[PanelGroupData], monthly_stats: dict) -> None:
        """Draw monthly comparison as grouped bar chart. @zara"""
        ax.set_facecolor(self.styles.background_light)

        months = list(range(1, 13))
        x = np.arange(len(months))
        n_groups = len(panel_groups)
        width = 0.8 / n_groups

        for i, group in enumerate(panel_groups):
            values = [monthly_stats[m][i].total_theoretical_kwh for m in months]
            offset = (i - n_groups/2 + 0.5) * width

            bars = ax.bar(
                x + offset,
                values,
                width,
                label=f"{group.name} ({group.power_kwp:.2f} kWp)",
                color=group.color,
                alpha=0.85,
                edgecolor=self.styles.border,
            )

            max_idx = values.index(max(values))
            ax.annotate(
                f"{values[max_idx]:.0f}",
                xy=(x[max_idx] + offset, values[max_idx]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8,
                color=group.color,
                fontweight="bold",
            )

        for season, months_list in [("winter", [0, 1, 11]), ("spring", [2, 3, 4]),
                                     ("summer", [5, 6, 7]), ("autumn", [8, 9, 10])]:
            for m in months_list:
                ax.axvspan(m - 0.5, m + 0.5, alpha=0.1, color=SEASON_COLORS[season])

        ax.set_xlabel("Monat", fontsize=11)
        ax.set_ylabel("Theoretische Produktion (kWh)", fontsize=11)
        ax.set_title("\ud83d\udcca Monatliche Produktion nach Panel-Gruppe", fontsize=14, fontweight="bold",
                     color=self.styles.text_primary, pad=10)

        ax.set_xticks(x)
        ax.set_xticklabels(MONTH_NAMES_SHORT_DE, fontsize=9)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_ylim(bottom=0)

    def _draw_seasonal_stacked(self, ax: "Axes", panel_groups: list[PanelGroupData], monthly_stats: dict) -> None:
        """Draw seasonal distribution as stacked bar chart. @zara"""
        ax.set_facecolor(self.styles.background_light)

        seasons = ["winter", "spring", "summer", "autumn"]
        season_months = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "autumn": [9, 10, 11],
        }

        x = np.arange(len(seasons))
        width = 0.6

        bottom = np.zeros(len(seasons))

        for i, group in enumerate(panel_groups):
            values = []
            for season in seasons:
                season_total = sum(
                    monthly_stats[m][i].total_theoretical_kwh
                    for m in season_months[season]
                    if m in monthly_stats
                )
                values.append(season_total)

            bars = ax.bar(
                x,
                values,
                width,
                bottom=bottom,
                label=group.name,
                color=group.color,
                alpha=0.85,
                edgecolor=self.styles.border,
            )

            for j, (val, b) in enumerate(zip(values, bottom)):
                if val > 50:
                    ax.text(
                        x[j], b + val/2,
                        f"{val:.0f}",
                        ha="center", va="center",
                        fontsize=10,
                        color="white" if i == 0 else "black",
                        fontweight="bold",
                    )

            bottom += values

        for j, total in enumerate(bottom):
            ax.text(
                x[j], total + 20,
                f"\u03a3 {total:.0f} kWh",
                ha="center", va="bottom",
                fontsize=11,
                color=self.styles.text_primary,
                fontweight="bold",
            )

        ax.set_xlabel("Jahreszeit", fontsize=11)
        ax.set_ylabel("Theoretische Produktion (kWh)", fontsize=11)
        ax.set_title("\ud83c\udf21\ufe0f Saisonale Verteilung (Stacked)", fontsize=14, fontweight="bold",
                     color=self.styles.text_primary, pad=10)

        ax.set_xticks(x)
        ax.set_xticklabels([SEASON_NAMES[s] for s in seasons], fontsize=11)
        ax.legend(loc="upper left", fontsize=10)
        ax.set_ylim(bottom=0, top=max(bottom) * 1.15)

    def _draw_daily_profile(self, ax: "Axes", group: PanelGroupData | None, monthly_stats: dict, group_idx: int) -> None:
        """Draw daily profile per season for a group. @zara"""
        ax.set_facecolor(self.styles.background_light)

        if group is None:
            ax.text(0.5, 0.5, "Keine Daten", transform=ax.transAxes,
                    ha="center", va="center", color=self.styles.text_muted)
            ax.set_title("Tagesverlauf", fontsize=12, color=self.styles.text_primary)
            return

        hourly_by_season = monthly_stats.get("hourly_by_season", {})
        hours = list(range(5, 21))

        seasons = ["summer", "spring", "autumn", "winter"]
        linestyles = ["-", "--", "-.", ":"]

        for season, ls in zip(seasons, linestyles):
            if season in hourly_by_season and group_idx in hourly_by_season[season]:
                hourly_data = hourly_by_season[season][group_idx]
                values = [np.mean(hourly_data.get(h, [0])) for h in hours]

                ax.plot(
                    hours, values,
                    label=SEASON_NAMES[season],
                    color=SEASON_COLORS[season],
                    linewidth=2.5,
                    linestyle=ls,
                    marker="o",
                    markersize=4,
                    alpha=0.9,
                )

        ax.fill_between(
            hours,
            [0] * len(hours),
            [np.mean(hourly_by_season.get("summer", {}).get(group_idx, {}).get(h, [0])) for h in hours],
            alpha=0.2,
            color=SEASON_COLORS["summer"],
        )

        ax.set_xlabel("Uhrzeit", fontsize=10)
        ax.set_ylabel("\u00d8 Produktion (kWh)", fontsize=10)
        ax.set_title(f"\ud83d\udd50 {group.name} - Tagesverlauf", fontsize=12, fontweight="bold",
                     color=group.color, pad=10)

        ax.set_xticks(range(6, 21, 2))
        ax.set_xticklabels([f"{h}:00" for h in range(6, 21, 2)], fontsize=9)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_xlim(5, 20)
        ax.set_ylim(bottom=0)

    def _draw_season_radar(self, ax: "Axes", panel_groups: list[PanelGroupData], monthly_stats: dict) -> None:
        """Draw radar chart for season comparison. @zara"""
        seasons = ["winter", "spring", "summer", "autumn"]
        season_months = {
            "winter": [12, 1, 2],
            "spring": [3, 4, 5],
            "summer": [6, 7, 8],
            "autumn": [9, 10, 11],
        }

        angles = np.linspace(0, 2 * np.pi, len(seasons), endpoint=False).tolist()
        angles += angles[:1]

        ax.set_facecolor(self.styles.background_light)

        for i, group in enumerate(panel_groups):
            values = []
            for season in seasons:
                season_total = sum(
                    monthly_stats[m][i].total_theoretical_kwh
                    for m in season_months[season]
                    if m in monthly_stats
                )
                values.append(season_total)

            max_val = max(values) if max(values) > 0 else 1
            values_norm = [v / max_val for v in values]
            values_norm += values_norm[:1]

            ax.plot(angles, values_norm, linewidth=2.5, linestyle="-",
                    label=group.name, color=group.color)
            ax.fill(angles, values_norm, alpha=0.25, color=group.color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([SEASON_NAMES[s] for s in seasons], fontsize=10)
        ax.set_ylim(0, 1.1)

        ax.set_title("\ud83c\udfaf Saison-Profil (normalisiert)", fontsize=12, fontweight="bold",
                     color=self.styles.text_primary, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    def _draw_efficiency_heatmap(self, ax: "Axes", panel_groups: list[PanelGroupData], monthly_stats: dict) -> None:
        """Draw efficiency heatmap (production per kWp). @zara"""
        ax.set_facecolor(self.styles.background_light)

        months = list(range(1, 13))
        n_groups = len(panel_groups)

        matrix = np.zeros((n_groups, len(months)))

        for i, group in enumerate(panel_groups):
            for j, m in enumerate(months):
                if m in monthly_stats and i < len(monthly_stats[m]):
                    kwh = monthly_stats[m][i].total_theoretical_kwh
                    kwp = group.power_kwp if group.power_kwp > 0 else 1
                    matrix[i, j] = kwh / kwp

        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "solar_cmap",
            [self.styles.background_light, "#FFA500", "#FFD700", "#FFFF00"],
            N=256
        )

        im = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )

        for i in range(n_groups):
            for j in range(len(months)):
                val = matrix[i, j]
                text_color = "black" if val > matrix.max() * 0.5 else "white"
                ax.text(
                    j, i,
                    f"{val:.0f}",
                    ha="center", va="center",
                    fontsize=9,
                    color=text_color,
                    fontweight="bold",
                )

        ax.set_xticks(np.arange(len(months)))
        ax.set_xticklabels(MONTH_NAMES_SHORT_DE, fontsize=9)
        ax.set_yticks(np.arange(n_groups))
        ax.set_yticklabels([g.name for g in panel_groups], fontsize=10)

        ax.set_xlabel("Monat", fontsize=10)
        ax.set_title("\u26a1 Effizienz (kWh/kWp)", fontsize=12, fontweight="bold",
                     color=self.styles.text_primary, pad=10)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("kWh pro kWp", fontsize=9)

    def _draw_footer(self, ax: "Axes", panel_groups: list[PanelGroupData]) -> None:
        """Draw footer with explanations. @zara"""
        ax.axis("off")
        ax.set_facecolor(self.styles.background)

        explanation = (
            "\ud83d\udccc Hinweise:\n"
            "\u2022 Die Werte zeigen die THEORETISCHE Maximalproduktion bei klarem Himmel\n"
            "\u2022 Tats\u00e4chliche Produktion kann durch Wolken, Verschattung, Temperatur etc. geringer sein\n"
            "\u2022 Gruppe 1 (S\u00fcd, 47\u00b0 Neigung): Optimal f\u00fcr Winter, da steiler Einfallswinkel\n"
            "\u2022 Gruppe 2 (SSW, 9\u00b0 Neigung): H\u00f6here Sommerproduktion durch flachere Ausrichtung\n"
        )

        ax.text(
            0.5, 0.7,
            explanation,
            transform=ax.transAxes,
            fontsize=10,
            color=self.styles.text_secondary,
            ha="center",
            va="top",
            linespacing=1.6,
            family="monospace",
        )

        ax.text(
            0.5, 0.1,
            f"Generiert: {datetime.now().strftime('%d.%m.%Y %H:%M')} | SFML Stats Panel-Gruppen Report",
            transform=ax.transAxes,
            fontsize=9,
            color=self.styles.text_muted,
            ha="center",
            va="bottom",
        )


async def generate_panel_seasonal_report(config_path: Path, output_path: Path | None = None) -> Path | None:
    """Standalone function to generate the report. @zara"""
    from ..storage import DataValidator

    validator = DataValidator(config_path)
    await validator.async_setup()

    chart = PanelSeasonalReportChart(validator)
    fig = await chart.generate()

    if output_path is None:
        output_path = chart.export_path / chart.get_filename()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def _save():
        import matplotlib.pyplot as plt
        fig.savefig(
            output_path,
            dpi=CHART_DPI,
            bbox_inches="tight",
            facecolor=chart.styles.background,
            edgecolor="none",
        )
        plt.close(fig)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _save)

    _LOGGER.info("Panel-Gruppen Report gespeichert: %s", output_path)
    return output_path
