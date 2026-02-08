# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Weather analytics export chart for SFML Stats. @zara"""
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


class WeatherAnalyticsChart:
    """Weather analytics PNG export chart. @zara"""

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
        title = f"Wetter Analytics - {period_names.get(self.period, self.period.capitalize())}"
        fig.suptitle(title, fontsize=20, fontweight='bold', color=self._styles.text_primary)

        ax_stats = fig.add_subplot(gs[0, :])
        self._render_stats_grid(ax_stats)

        ax_temp = fig.add_subplot(gs[1, 0])
        self._render_temperature_chart(ax_temp)

        ax_rad = fig.add_subplot(gs[1, 1])
        self._render_radiation_chart(ax_rad)

        ax_rain = fig.add_subplot(gs[2, 0])
        self._render_rain_chart(ax_rain)

        ax_wind = fig.add_subplot(gs[2, 1])
        self._render_wind_chart(ax_wind)

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
            f"\u00d8 Temperatur (Woche): {self.stats.get('avgTemp', 0):.1f}\u00b0C",
            f"H\u00f6chsttemperatur: {self.stats.get('maxTemp', 0):.1f}\u00b0C",
            f"Tiefsttemperatur: {self.stats.get('minTemp', 0):.1f}\u00b0C",
            f"Niederschlag (Monat): {self.stats.get('totalRain', 0):.1f} mm",
            f"\u00d8 Wind: {self.stats.get('avgWind', 0):.1f} m/s",
            f"Sonnenstunden (Monat): {self.stats.get('sunHours', 0):.0f} h",
        ]

        positions = [
            (0.17, 0.7), (0.5, 0.7), (0.83, 0.7),
            (0.17, 0.3), (0.5, 0.3), (0.83, 0.3),
        ]

        for text, (x, y) in zip(stats_text, positions):
            props = dict(
                boxstyle='round,pad=0.8',
                facecolor=self._styles.background_card,
                edgecolor='#f59e0b',
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

    def _render_temperature_chart(self, ax: Any) -> None:
        """Render temperature timeline. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verf\u00fcgbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        temp_avg = [d.get('temp_avg', 0) for d in data_reversed]
        temp_max = [d.get('temp_max', 0) for d in data_reversed]
        temp_min = [d.get('temp_min', 0) for d in data_reversed]

        marker_size = 6 if len(dates) <= 30 else 4 if len(dates) <= 100 else 2

        ax.plot(dates, temp_avg, marker='o', linewidth=2, markersize=marker_size,
                color='#f59e0b', label='Durchschnitt')
        ax.plot(dates, temp_max, linewidth=1, linestyle='--',
                color='#ef4444', label='Maximum', alpha=0.7)
        ax.plot(dates, temp_min, linewidth=1, linestyle='--',
                color='#3b82f6', label='Minimum', alpha=0.7)

        ax.fill_between(range(len(dates)), temp_min, temp_max, alpha=0.1, color='#f59e0b')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Temperatur (\u00b0C)', color=self._styles.text_primary)
        ax.set_title('Temperatur Verlauf', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.legend(facecolor=self._styles.background_card, edgecolor=self._styles.border)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

    def _render_radiation_chart(self, ax: Any) -> None:
        """Render radiation with solar correlation. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verf\u00fcgbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        radiation = [d.get('radiation', d.get('radiation_avg', 0)) for d in data_reversed]
        solar = [d.get('solar_kwh', 0) for d in data_reversed]

        ax2 = ax.twinx()
        ax.bar(range(len(dates)), radiation, alpha=0.8, color='#fbbf24', label='Einstrahlung')
        ax2.plot(dates, solar, marker='o', markersize=4, linewidth=2,
                 color='#22c55e', label='Solar-Produktion')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Einstrahlung (W/m\u00b2)', color=self._styles.text_primary)
        ax2.set_ylabel('Solar-Produktion (kWh)', color=self._styles.text_primary)
        ax.set_title('Einstrahlung & Solar-Korrelation', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax2.tick_params(colors=self._styles.text_secondary)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _render_rain_chart(self, ax: Any) -> None:
        """Render rain and humidity. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verf\u00fcgbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        rain = [d.get('rain', d.get('rain_total', 0)) for d in data_reversed]
        humidity = [d.get('humidity', d.get('humidity_avg', 0)) for d in data_reversed]

        ax2 = ax.twinx()
        ax.bar(range(len(dates)), rain, alpha=0.8, color='#3b82f6', label='Niederschlag')
        ax2.plot(dates, humidity, marker='o', markersize=4, linewidth=2,
                 color='#06b6d4', label='Luftfeuchtigkeit')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Niederschlag (mm)', color=self._styles.text_primary)
        ax2.set_ylabel('Luftfeuchtigkeit (%)', color=self._styles.text_primary)
        ax2.set_ylim(0, 100)
        ax.set_title('Niederschlag & Luftfeuchtigkeit', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax2.tick_params(colors=self._styles.text_secondary)
        ax.grid(True, alpha=0.2, color=self._styles.grid, axis='y')
        ax.set_facecolor(self._styles.background)

    def _render_wind_chart(self, ax: Any) -> None:
        """Render wind speed chart. @zara"""
        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verf\u00fcgbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        data_reversed = list(reversed(self.data))
        dates = [d['date'][5:] for d in data_reversed]
        wind = [d.get('wind', d.get('wind_avg', 0)) for d in data_reversed]

        ax.plot(dates, wind, marker='o', markersize=4, linewidth=2,
                color='#a855f7', label='Wind')
        ax.fill_between(range(len(dates)), wind, alpha=0.2, color='#a855f7')

        ax.set_xlabel('Datum', color=self._styles.text_primary)
        ax.set_ylabel('Wind (m/s)', color=self._styles.text_primary)
        ax.set_title('Wind Geschwindigkeit', color=self._styles.text_primary, fontweight='bold')

        self._apply_adaptive_ticks(ax, dates)
        ax.tick_params(colors=self._styles.text_secondary)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
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

    def _add_footer(self, fig: Figure) -> None:
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
            "SFML Stats - Wetter Analytics",
            fontsize=9,
            color=self._styles.text_muted,
            ha='left',
            va='bottom',
            transform=fig.transFigure,
            style='italic',
        )
