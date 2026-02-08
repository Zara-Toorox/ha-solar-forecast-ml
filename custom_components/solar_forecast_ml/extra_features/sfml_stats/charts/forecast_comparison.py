# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Forecast comparison chart for SFML Stats. @zara"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .base import BaseChart
from .styles import ChartStyles, apply_dark_theme
from ..const import FORECAST_COMPARISON_CHART_DAYS
from ..readers.forecast_comparison_reader import ForecastComparisonReader

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from ..storage import DataValidator

_LOGGER = logging.getLogger(__name__)


class ForecastComparisonChart(BaseChart):
    """7-day forecast comparison chart. @zara"""

    def __init__(
        self,
        validator: "DataValidator",
        figsize: tuple[int, int] = (14, 8),
    ) -> None:
        """Initialize the chart. @zara"""
        super().__init__(validator, figsize)
        self._reader = ForecastComparisonReader(validator.config_path)

    def get_filename(self, **kwargs: Any) -> str:
        """Get filename for the chart. @zara"""
        today = date.today().isoformat()
        return f"forecast_comparison_{today}.png"

    async def generate(self, **kwargs: Any) -> "Figure":
        """Generate the forecast comparison chart. @zara"""
        days = kwargs.get("days", FORECAST_COMPARISON_CHART_DAYS)

        chart_data = await self._reader.async_get_chart_data(days=days)

        fig = await self._run_in_executor(
            self._generate_sync,
            chart_data,
        )

        self._fig = fig
        return fig

    def _generate_sync(self, data: dict[str, Any]) -> "Figure":
        """Synchronous chart generation in executor. @zara"""
        import matplotlib.pyplot as plt
        import numpy as np

        apply_dark_theme()

        fig, ax = self._create_figure(figsize=self._figsize)
        ax.set_facecolor(self._styles.background_light)

        dates = data["dates"]
        actual = data["actual"]
        sfml = data["sfml"]
        external_1 = data.get("external_1")
        external_1_name = data.get("external_1_name", "Extern 1")
        external_2 = data.get("external_2")
        external_2_name = data.get("external_2_name", "Extern 2")
        stats = data.get("stats", {})

        x = np.arange(len(dates))

        actual_clean = [v if v is not None else np.nan for v in actual]
        ax.plot(
            x, actual_clean,
            label="Tatsächlicher Ertrag",
            color=self._styles.actual,
            linewidth=2.5,
            marker="o",
            markersize=6,
            alpha=0.95,
            zorder=10,
        )

        sfml_clean = [v if v is not None else np.nan for v in sfml]
        ax.plot(
            x, sfml_clean,
            label="SFML Prognose",
            color=self._styles.ml_purple,
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=5,
            alpha=0.85,
            zorder=9,
        )

        if external_1 is not None:
            ext1_clean = [v if v is not None else np.nan for v in external_1]
            if any(v is not None for v in external_1):
                ax.plot(
                    x, ext1_clean,
                    label=external_1_name,
                    color=self._styles.neon_cyan,
                    linewidth=2.0,
                    linestyle=":",
                    marker="^",
                    markersize=5,
                    alpha=0.8,
                    zorder=8,
                )

        if external_2 is not None:
            ext2_clean = [v if v is not None else np.nan for v in external_2]
            if any(v is not None for v in external_2):
                ax.plot(
                    x, ext2_clean,
                    label=external_2_name,
                    color=self._styles.solar_orange,
                    linewidth=2.0,
                    linestyle="-.",
                    marker="d",
                    markersize=5,
                    alpha=0.8,
                    zorder=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(dates, rotation=45, ha="right")

        ax.set_xlabel("Datum", fontsize=self._styles.label_size)
        ax.set_ylabel("Energie (kWh)", fontsize=self._styles.label_size)

        self._add_title(
            ax,
            "7-Tage Prognose-Vergleich",
            "Vergleich der Prognosegenauigkeit verschiedener Quellen",
        )

        ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.xaxis.grid(False)

        all_values = []
        for series in [actual_clean, sfml_clean]:
            all_values.extend([v for v in series if not np.isnan(v)])
        if external_1:
            ext1_clean = [v if v is not None else np.nan for v in external_1]
            all_values.extend([v for v in ext1_clean if not np.isnan(v)])
        if external_2:
            ext2_clean = [v if v is not None else np.nan for v in external_2]
            all_values.extend([v for v in ext2_clean if not np.isnan(v)])

        if all_values:
            ax.set_ylim(bottom=0, top=max(all_values) * 1.15)
        else:
            ax.set_ylim(bottom=0, top=50)

        ax.legend(
            loc="upper left",
            fontsize=self._styles.legend_size,
            framealpha=0.9,
            edgecolor=self._styles.border,
            facecolor=self._styles.background_card,
        )

        self._add_stats_box(ax, stats, external_1_name, external_2_name)

        self._add_footer(fig, "Prognose-Vergleich")

        plt.tight_layout()

        return fig

    def _add_stats_box(
        self,
        ax: Any,
        stats: dict[str, Any],
        ext1_name: str,
        ext2_name: str,
    ) -> None:
        """Add accuracy statistics box to the chart. @zara"""
        lines = []

        sfml_acc = stats.get("sfml_avg_accuracy")
        if sfml_acc is not None:
            lines.append(f"SFML: {sfml_acc:.1f}%")

        ext1_acc = stats.get("external_1_avg_accuracy")
        if ext1_acc is not None:
            lines.append(f"{ext1_name}: {ext1_acc:.1f}%")

        ext2_acc = stats.get("external_2_avg_accuracy")
        if ext2_acc is not None:
            lines.append(f"{ext2_name}: {ext2_acc:.1f}%")

        best = stats.get("best_forecast")
        if best:
            lines.append(f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500")
            lines.append(f"Beste: {best}")

        if not lines:
            lines.append("Keine Daten")

        text = "\u00d8 Genauigkeit\n" + "\n".join(lines)

        props = dict(
            boxstyle="round,pad=0.5",
            facecolor=self._styles.background_card,
            edgecolor=self._styles.border,
            alpha=0.9,
        )

        ax.text(
            0.98, 0.98,
            text,
            transform=ax.transAxes,
            fontsize=self._styles.label_size,
            color=self._styles.text_primary,
            ha="right",
            va="top",
            bbox=props,
            family="monospace",
        )
