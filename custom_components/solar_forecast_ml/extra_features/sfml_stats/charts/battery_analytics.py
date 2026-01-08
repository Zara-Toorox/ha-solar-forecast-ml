# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Battery Analytics Export Chart for SFML Stats."""
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

# Shared executor for matplotlib operations
_MATPLOTLIB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="matplotlib")


class BatteryAnalyticsChart:
    """Chart für Battery Analytics PNG-Export."""

    def __init__(
        self,
        period: str,
        stats: dict[str, Any],
        data: list[dict[str, Any]],
    ) -> None:
        """Initialize chart.

        Args:
            period: 'week', 'month', or 'year'
            stats: Statistics dict with weekCharged, weekDischarged, etc.
            data: Historical data points
        """
        self.period = period
        self.stats = stats
        self.data = self._filter_data_by_period(data, period)
        self._styles = ChartStyles()

    def _filter_data_by_period(self, data: list[dict[str, Any]], period: str) -> list[dict[str, Any]]:
        """Filter data based on selected period."""
        if not data:
            return []

        sorted_data = sorted(data, key=lambda x: x.get('date', ''), reverse=True)

        days_map = {
            'week': 7,
            'month': 30,
            'year': 365
        }
        days = days_map.get(period, 7)

        return sorted_data[:days]

    async def async_render(self) -> bytes:
        """Render chart to PNG bytes."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _MATPLOTLIB_EXECUTOR,
            self._render_sync
        )

    def _render_sync(self) -> bytes:
        """Synchronous render - runs in executor thread."""
        import matplotlib.pyplot as plt
        from .styles import apply_dark_theme

        apply_dark_theme()

        fig = plt.figure(figsize=(14, 10), facecolor=self._styles.background)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3, top=0.92, bottom=0.08)

        # Title
        period_names = {'week': 'Letzte 7 Tage', 'month': 'Letzter Monat', 'year': 'Dieses Jahr'}
        title = f"Battery Analytics - {period_names.get(self.period, self.period.capitalize())}"
        fig.suptitle(title, fontsize=20, fontweight='bold', color=self._styles.text_primary)

        # Stats Grid (top row, full width)
        ax_stats = fig.add_subplot(gs[0, :])
        self._render_stats_grid(ax_stats)

        # SOC Chart (middle left)
        ax_soc = fig.add_subplot(gs[1, 0])
        self._render_soc_chart(ax_soc)

        # Charge/Discharge Chart (middle right)
        ax_charge = fig.add_subplot(gs[1, 1])
        self._render_charge_chart(ax_charge)

        # Efficiency Chart (bottom left)
        ax_efficiency = fig.add_subplot(gs[2, 0])
        self._render_efficiency_chart(ax_efficiency)

        # Power Distribution (bottom right)
        ax_power = fig.add_subplot(gs[2, 1])
        self._render_power_chart(ax_power)

        # Footer
        self._add_footer(fig)

        # Render to bytes
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
        """Render stats as a grid."""
        ax.axis('off')

        stats_text = [
            f"Geladen (Mo-So): {self.stats.get('weekCharged', 0):.2f} kWh",
            f"Entladen (Mo-So): {self.stats.get('weekDischarged', 0):.2f} kWh",
            f"Ø SOC: {self.stats.get('avgSOC', 0):.0f}%",
            f"Wirkungsgrad: {self.stats.get('efficiency', 0):.1f}%",
            f"Zyklen (Mo-So): {self.stats.get('cycles', 0):.1f}",
            f"Peak Leistung: {self.stats.get('peakPower', 0):.0f} W",
        ]

        positions = [
            (0.17, 0.7), (0.5, 0.7), (0.83, 0.7),
            (0.17, 0.3), (0.5, 0.3), (0.83, 0.3),
        ]

        for text, (x, y) in zip(stats_text, positions):
            props = dict(
                boxstyle='round,pad=0.8',
                facecolor=self._styles.background_card,
                edgecolor='#22c55e',  # Grün wie Battery Node
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

    def _render_soc_chart(self, ax: Any) -> None:
        """Render SOC timeline."""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        soc = [d.get('soc', 0) for d in data_reversed]

        marker_size = 6 if len(dates) <= 30 else 4 if len(dates) <= 100 else 2

        ax.plot(dates, soc, marker='o', linewidth=2, markersize=marker_size,
                color=self._styles.accuracy_good, label='SOC')
        ax.fill_between(range(len(dates)), soc, alpha=0.2, color=self._styles.accuracy_good)

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('SOC (%)', color=self._styles.text_primary)
        ax.set_title('Batterieladezustand (SOC)', color=self._styles.text_primary, fontweight='bold')
        ax.set_ylim(0, 100)

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

    def _render_charge_chart(self, ax: Any) -> None:
        """Render charge/discharge chart."""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        charged = [d.get('charged_kwh', 0) for d in data_reversed]
        discharged = [-d.get('discharged_kwh', 0) for d in data_reversed]

        x = range(len(dates))
        width = 0.8

        ax.bar(x, charged, width, label='Geladen', color=self._styles.accuracy_good, alpha=0.9)
        ax.bar(x, discharged, width, label='Entladen', color=self._styles.accuracy_medium, alpha=0.9)

        ax.axhline(y=0, color=self._styles.text_secondary, linestyle='-', linewidth=0.8)
        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Energie (kWh)', color=self._styles.text_primary)
        ax.set_title('Laden / Entladen', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _render_efficiency_chart(self, ax: Any) -> None:
        """Render efficiency timeline."""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        # Validiere Wirkungsgrad-Werte auf sinnvollen Bereich (0-100%)
        efficiency = [max(0, min(100, d.get('efficiency', 0))) for d in data_reversed]

        avg = sum(efficiency) / len(efficiency) if efficiency else 0
        marker_size = 6 if len(dates) <= 30 else 4 if len(dates) <= 100 else 2

        ax.plot(dates, efficiency, marker='o', linewidth=2, markersize=marker_size,
                color=self._styles.predicted, label='Wirkungsgrad')
        ax.axhline(y=avg, color=self._styles.solar_orange, linestyle='--', linewidth=1.5,
                   label=f'Ø {avg:.1f}%')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Wirkungsgrad (%)', color=self._styles.text_primary)
        ax.set_title('Batterie-Wirkungsgrad', color=self._styles.text_primary, fontweight='bold')
        ax.set_ylim(0, 100)

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

    def _render_power_chart(self, ax: Any) -> None:
        """Render power distribution."""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        power = [d.get('peak_power_w', 0) for d in data_reversed]

        colors = [self._styles.accuracy_good if p > 0 else self._styles.text_muted for p in power]
        ax.bar(range(len(dates)), power, color=colors, alpha=0.8)

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Peak Leistung (W)', color=self._styles.text_primary)
        ax.set_title('Maximale Leistung pro Tag', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _apply_adaptive_ticks(self, ax: Any, dates: list[str]) -> None:
        """Apply adaptive tick labels based on data length."""
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

    def _add_footer(self, fig: "Figure") -> None:
        """Add footer with timestamp."""
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
            "SFML Stats - Battery Analytics",
            fontsize=9,
            color=self._styles.text_muted,
            ha='left',
            va='bottom',
            transform=fig.transFigure,
            style='italic',
        )
