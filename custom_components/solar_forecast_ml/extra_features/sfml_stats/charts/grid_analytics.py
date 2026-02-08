# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Grid analytics export chart for SFML Stats. @zara"""
from __future__ import annotations

import asyncio
import io
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .styles import ChartStyles
from ..const import CHART_DPI

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_LOGGER = logging.getLogger(__name__)

_MATPLOTLIB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="matplotlib")


class GridAnalyticsChart:
    """Grid analytics PNG export chart. @zara"""

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
        title = f"Netz Analytics - {period_names.get(self.period, self.period.capitalize())}"
        fig.suptitle(title, fontsize=20, fontweight='bold', color=self._styles.text_primary)

        ax_stats = fig.add_subplot(gs[0, :])
        self._render_stats_grid(ax_stats)

        ax_flow = fig.add_subplot(gs[1, 0])
        self._render_flow_chart(ax_flow)

        ax_price = fig.add_subplot(gs[1, 1])
        self._render_price_chart(ax_price)

        ax_money = fig.add_subplot(gs[2, 0])
        self._render_money_chart(ax_money)

        ax_pattern = fig.add_subplot(gs[2, 1])
        self._render_pattern_chart(ax_pattern)

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
            f"Bezug (Mo-So): {self.stats.get('weekImport', 0):.2f} kWh",
            f"Einspeisung (Mo-So): {self.stats.get('weekExport', 0):.2f} kWh",
            f"Netto Balance: {self.stats.get('netBalance', 0):.2f} kWh",
            f"Ø Strompreis: {self.stats.get('avgPrice', 0):.2f} ct/kWh",
            f"Kosten (Mo-So): {self.stats.get('costs', 0):.2f} \u20ac",
            f"Erl\u00f6se (Mo-So): {self.stats.get('revenue', 0):.2f} \u20ac",
        ]

        positions = [
            (0.17, 0.7), (0.5, 0.7), (0.83, 0.7),
            (0.17, 0.3), (0.5, 0.3), (0.83, 0.3),
        ]

        for text, (x, y) in zip(stats_text, positions):
            props = dict(
                boxstyle='round,pad=0.8',
                facecolor=self._styles.background_card,
                edgecolor='#8b5cf6',
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

    def _render_flow_chart(self, ax: Any) -> None:
        """Render import/export chart. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        import_kwh = [d.get('import_kwh', 0) for d in data_reversed]
        export_kwh = [-d.get('export_kwh', 0) for d in data_reversed]

        x = range(len(dates))
        width = 0.8

        ax.bar(x, import_kwh, width, label='Bezug', color=self._styles.price_red, alpha=0.9)
        ax.bar(x, export_kwh, width, label='Einspeisung', color=self._styles.price_green, alpha=0.9)

        ax.axhline(y=0, color=self._styles.text_secondary, linestyle='-', linewidth=0.8)
        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Energie (kWh)', color=self._styles.text_primary)
        ax.set_title('Netzbezug / Einspeisung', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _render_price_chart(self, ax: Any) -> None:
        """Render price timeline. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        prices = [d.get('avg_price_ct', 0) for d in data_reversed]

        if not prices or all(p == 0 for p in prices):
            ax.text(0.5, 0.5, 'Keine Preisdaten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        avg_price = sum(prices) / len(prices) if prices else 0
        marker_size = 6 if len(dates) <= 30 else 4 if len(dates) <= 100 else 2

        colors = [self._styles.price_green if p < avg_price * 0.9
                  else self._styles.price_red if p > avg_price * 1.1
                  else self._styles.solar_orange for p in prices]

        ax.scatter(range(len(dates)), prices, c=colors, s=marker_size*10, alpha=0.6)
        ax.plot(dates, prices, linewidth=2, color=self._styles.predicted, alpha=0.3)
        ax.axhline(y=avg_price, color=self._styles.solar_yellow, linestyle='--', linewidth=1.5,
                   label=f'\u00d8 {avg_price:.2f} ct/kWh')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Strompreis (ct/kWh)', color=self._styles.text_primary)
        ax.set_title('Strompreis Verlauf', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

    def _render_money_chart(self, ax: Any) -> None:
        """Render cost/revenue analysis. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        costs = [d.get('costs_eur', 0) for d in data_reversed]
        revenue = [d.get('revenue_eur', 0) for d in data_reversed]

        x = np.arange(len(dates))
        width = 0.4

        ax.bar(x - width/2, costs, width, label='Kosten', color=self._styles.price_red, alpha=0.9)
        ax.bar(x + width/2, revenue, width, label='Erl\u00f6se', color=self._styles.price_green, alpha=0.9)

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Betrag (\u20ac)', color=self._styles.text_primary)
        ax.set_title('Kosten & Erl\u00f6se', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _render_pattern_chart(self, ax: Any) -> None:
        """Render usage patterns. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        net_flow = [d.get('import_kwh', 0) - d.get('export_kwh', 0) for d in data_reversed]

        colors = [self._styles.price_red if flow > 0 else self._styles.price_green
                  for flow in net_flow]
        ax.bar(range(len(dates)), net_flow, color=colors, alpha=0.8)

        ax.axhline(y=0, color=self._styles.text_secondary, linestyle='-', linewidth=0.8)
        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Netto-Energiefluss (kWh)', color=self._styles.text_primary)
        ax.set_title('Nutzungsmuster (+ Bezug / - Einspeisung)', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _apply_adaptive_ticks(self, ax: Any, dates: list[str]) -> None:
        """Apply adaptive tick labels based on data length. @zara"""
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
            "SFML Stats - Netz Analytics",
            fontsize=9,
            color=self._styles.text_muted,
            ha='left',
            va='bottom',
            transform=fig.transFigure,
            style='italic',
        )
