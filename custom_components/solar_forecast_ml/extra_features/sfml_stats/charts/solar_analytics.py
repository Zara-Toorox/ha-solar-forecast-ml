# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Solar analytics export chart for SFML Stats. @zara"""
from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .styles import ChartStyles
from ..const import CHART_DPI

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_LOGGER = logging.getLogger(__name__)

_MATPLOTLIB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="matplotlib")


class SolarAnalyticsChart:
    """Solar analytics PNG export chart. @zara"""

    def __init__(
        self,
        period: str,
        stats: dict[str, Any],
        data: list[dict[str, Any]],
    ) -> None:
        """Initialize chart. @zara"""
        self.period = period
        self.stats = stats
        self.data = self._filter_data_by_period(data, period)
        self._styles = ChartStyles()

    def _filter_data_by_period(self, data: list[dict[str, Any]], period: str) -> list[dict[str, Any]]:
        """Filter data based on selected period. @zara"""
        if not data:
            return []

        from datetime import datetime, timedelta

        sorted_data = sorted(data, key=lambda x: x.get('date', ''), reverse=True)

        days_map = {
            'week': 7,
            'month': 30,
            'year': 365
        }
        days = days_map.get(period, 7)

        return sorted_data[:days]

    async def async_render(self) -> bytes:
        """Render chart to PNG bytes. @zara"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _MATPLOTLIB_EXECUTOR,
            self._render_sync
        )

    def _render_sync(self) -> bytes:
        """Synchronous render in executor thread. @zara"""
        import matplotlib.pyplot as plt
        from .styles import apply_dark_theme

        apply_dark_theme()

        fig = plt.figure(figsize=(14, 10), facecolor=self._styles.background)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, top=0.92, bottom=0.08)

        period_names = {'week': 'Letzte 7 Tage', 'month': 'Letzter Monat', 'year': 'Dieses Jahr'}
        title = f"Solar Analytics - {period_names.get(self.period, self.period.capitalize())}"
        fig.suptitle(title, fontsize=20, fontweight='bold', color=self._styles.text_primary)

        ax_stats = fig.add_subplot(gs[0, :])
        self._render_stats_grid(ax_stats)

        ax_production = fig.add_subplot(gs[1, 0])
        self._render_production_chart(ax_production)

        ax_accuracy = fig.add_subplot(gs[1, 1])
        self._render_accuracy_chart(ax_accuracy)

        ax_heatmap = fig.add_subplot(gs[2, :])
        self._render_peak_heatmap(ax_heatmap)

        self._add_footer(fig)

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            dpi=CHART_DPI,
            bbox_inches='tight',
            facecolor=self._styles.background,
            edgecolor='none',
        )
        plt.close(fig)

        buf.seek(0)
        return buf.read()

    def _render_stats_grid(self, ax: Any) -> None:
        """Render stats as a grid. @zara"""
        ax.axis('off')

        stats_text = [
            f"Diese Woche: {self.stats.get('weekTotal', 0):.2f} kWh",
            f"Dieser Monat: {self.stats.get('monthTotal', 0):.2f} kWh",
            f"Dieses Jahr: {self.stats.get('yearTotal', 0):.2f} kWh",
            f"\u00d8 Genauigkeit: {self.stats.get('avgAccuracy', 0):.1f}%",
            f"Peak Leistung: {self.stats.get('peakPower', 0):.0f} W",
            f"\u00d8 pro Tag: {self.stats.get('avgDaily', 0):.2f} kWh",
        ]

        positions = [
            (0.17, 0.7), (0.5, 0.7), (0.83, 0.7),
            (0.17, 0.3), (0.5, 0.3), (0.83, 0.3),
        ]

        for text, (x, y) in zip(stats_text, positions):
            props = dict(
                boxstyle='round,pad=0.8',
                facecolor=self._styles.background_card,
                edgecolor=self._styles.solar_yellow,
                linewidth=2,
                alpha=0.9
            )
            ax.text(
                x, y,
                text,
                transform=ax.transAxes,
                fontsize=12,
                fontweight='bold',
                color=self._styles.text_primary,
                ha='center',
                va='center',
                bbox=props
            )

    def _render_production_chart(self, ax: Any) -> None:
        """Render production vs prediction chart. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verf\u00fcgbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))

        dates = [d['date'][5:] for d in data_reversed]
        actual = [d.get('actual_kwh', 0) for d in data_reversed]
        predicted = [d.get('predicted_kwh', 0) for d in data_reversed]

        x = range(len(dates))
        width = 0.4

        ax.bar([i - width/2 for i in x], actual, width, label='Tats\u00e4chlich',
               color=self._styles.actual, alpha=0.9, edgecolor='black')
        ax.bar([i + width/2 for i in x], predicted, width, label='Prognose',
               color=self._styles.predicted, alpha=0.7, edgecolor='black')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Produktion (kWh)', color=self._styles.text_primary)
        ax.set_title('Produktion vs. Prognose', color=self._styles.text_primary, fontweight='bold')

        if len(dates) > 60:
            tick_positions = list(range(0, len(dates), 14))
            tick_labels = [dates[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        elif len(dates) > 20:
            tick_positions = list(range(0, len(dates), 3))
            tick_labels = [dates[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(dates, rotation=45, ha='right')

        ax.tick_params(colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

    def _render_accuracy_chart(self, ax: Any) -> None:
        """Render accuracy timeline. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verf\u00fcgbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))

        dates = [d['date'][5:] for d in data_reversed]
        accuracy = [d.get('accuracy', 0) for d in data_reversed]

        marker_size = 6 if len(dates) <= 30 else 4 if len(dates) <= 100 else 2

        ax.plot(dates, accuracy, marker='o', linewidth=2, markersize=marker_size,
                color=self._styles.accuracy_good, label='Genauigkeit')
        ax.axhline(y=sum(accuracy)/len(accuracy) if accuracy else 0, color=self._styles.predicted,
                   linestyle='--', linewidth=1.5, label='Durchschnitt')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Genauigkeit (%)', color=self._styles.text_primary)
        ax.set_title('Prognose-Genauigkeit', color=self._styles.text_primary, fontweight='bold')
        ax.set_ylim(0, 100)

        if len(dates) > 60:
            tick_positions = list(range(0, len(dates), 14))
            tick_labels = [dates[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        elif len(dates) > 20:
            tick_positions = list(range(0, len(dates), 3))
            tick_labels = [dates[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha='right')

        ax.tick_params(axis='y', colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

    def _render_peak_heatmap(self, ax: Any) -> None:
        """Render peak times heatmap. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verf\u00fcgbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))

        dates = [d['date'][5:] for d in data_reversed]
        peaks = [d.get('peak_power_w', 0) for d in data_reversed]

        colors = [self._styles.solar_yellow if p > 0 else self._styles.text_muted for p in peaks]
        ax.bar(range(len(dates)), peaks, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Peak Leistung (W)', color=self._styles.text_primary)
        ax.set_title('Peak-Leistung pro Tag', color=self._styles.text_primary, fontweight='bold')

        if len(dates) > 60:
            tick_positions = list(range(0, len(dates), 14))
            tick_labels = [dates[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        elif len(dates) > 20:
            tick_positions = list(range(0, len(dates), 3))
            tick_labels = [dates[i] for i in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        else:
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels(dates, rotation=45, ha='right')

        ax.tick_params(axis='y', colors=self._styles.text_secondary)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _add_footer(self, fig: "Figure") -> None:
        """Add footer with timestamp. @zara"""
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
        footer_text = f"Generiert: {timestamp}"

        fig.text(
            0.99, 0.02,
            footer_text,
            fontsize=9,
            color=self._styles.text_muted,
            ha='right',
            va='bottom',
            transform=fig.transFigure,
        )

        fig.text(
            0.01, 0.02,
            "SFML Stats - Solar Analytics",
            fontsize=9,
            color=self._styles.text_muted,
            ha='left',
            va='bottom',
            transform=fig.transFigure,
            style='italic',
        )
