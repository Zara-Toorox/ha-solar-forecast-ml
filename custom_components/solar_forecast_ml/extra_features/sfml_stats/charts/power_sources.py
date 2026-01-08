# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Power Sources Export Chart for SFML Stats."""
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


class PowerSourcesChart:
    """Chart für Power Sources PNG-Export - Stacked Area Chart. @zara"""

    def __init__(
        self,
        period: str,
        stats: dict[str, Any],
        data: list[dict[str, Any]],
    ) -> None:
        """Initialize chart. @zara

        Args:
            period: 'today', 'week', or 'custom'
            stats: Statistics dict with totals, averages, etc.
            data: Time series data points with timestamps and power values
        """
        self.period = period
        self.stats = stats
        self.data = data
        self._styles = ChartStyles()

    async def async_render(self) -> bytes:
        """Render chart to PNG bytes. @zara

        Runs all matplotlib operations in a thread executor to avoid
        blocking the event loop.

        Returns:
            PNG image as bytes
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _MATPLOTLIB_EXECUTOR,
            self._render_sync
        )

    def _render_sync(self) -> bytes:
        """Synchronous render - runs in executor thread. @zara"""
        import matplotlib.pyplot as plt
        import numpy as np
        from .styles import apply_dark_theme

        try:
            _LOGGER.debug("Rendering power sources chart: period=%s, data_points=%d",
                          self.period, len(self.data) if self.data else 0)
        except Exception:
            pass

        apply_dark_theme()

        # Create figure with 2 rows
        fig = plt.figure(figsize=(16, 10), facecolor=self._styles.background)
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.15, top=0.92, bottom=0.08)

        # Title
        period_names = {
            'today': 'Heute',
            'week': 'Letzte 7 Tage',
            'custom': 'Benutzerdefiniert'
        }
        title = f"Power Sources - {period_names.get(self.period, self.period.capitalize())}"
        fig.suptitle(title, fontsize=20, fontweight='bold', color=self._styles.text_primary)

        # Main stacked area chart
        ax_main = fig.add_subplot(gs[0])
        self._render_stacked_area(ax_main)

        # Battery SOC chart (bottom)
        ax_soc = fig.add_subplot(gs[1], sharex=ax_main)
        self._render_battery_soc(ax_soc)

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

    def _render_stacked_area(self, ax: Any) -> None:
        """Render the main stacked area chart. @zara"""
        import matplotlib.pyplot as plt
        import numpy as np

        if not self.data:
            ax.text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        # Extract data
        timestamps = []
        solar_to_house = []
        solar_to_battery = []
        battery = []
        grid = []
        consumption = []

        for point in self.data:
            ts = point.get('timestamp', '')
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    timestamps.append(dt)
                except (ValueError, TypeError):
                    continue

                # Get values, default to 0 if None
                # Values are in W, convert to kW for display
                solar_to_house_val = (point.get('solar_to_house') or 0) / 1000
                solar_to_battery_val = (point.get('solar_to_battery') or 0) / 1000
                battery_val = (point.get('battery_to_house') or 0) / 1000
                grid_val = (point.get('grid_to_house') or 0) / 1000
                consumption_val = (point.get('home_consumption') or 0) / 1000

                # Ensure non-negative
                solar_to_house.append(max(0, solar_to_house_val))
                solar_to_battery.append(max(0, solar_to_battery_val))
                battery.append(max(0, battery_val))
                grid.append(max(0, grid_val))
                consumption.append(max(0, consumption_val))

        if not timestamps:
            ax.text(0.5, 0.5, 'Keine gültigen Zeitstempel', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=14)
            ax.axis('off')
            return

        # Convert to numpy arrays
        solar_to_house = np.array(solar_to_house)
        solar_to_battery = np.array(solar_to_battery)
        battery = np.array(battery)
        grid = np.array(grid)
        consumption = np.array(consumption)

        # Colors matching the frontend
        solar_to_house_color = '#FFB74D'  # Orange for Solar → Haus
        solar_to_battery_color = '#9ACD32'  # Yellow-Green for Solar → Batterie
        battery_color = '#4DD0E1'  # Cyan for Batterie → Haus
        grid_color = '#90CAF9'  # Light Blue for Netz → Haus

        # Create stacked area chart for house consumption sources
        ax.fill_between(timestamps, 0, solar_to_house,
                        color=solar_to_house_color, alpha=0.8, label='Solar → Haus')
        ax.fill_between(timestamps, solar_to_house, solar_to_house + battery,
                        color=battery_color, alpha=0.8, label='Batterie → Haus')
        ax.fill_between(timestamps, solar_to_house + battery, solar_to_house + battery + grid,
                        color=grid_color, alpha=0.8, label='Netz → Haus')

        # Solar to Battery as separate line (not stacked, as it doesn't go to house)
        ax.plot(timestamps, solar_to_battery, color=solar_to_battery_color, linewidth=2,
                label='Solar → Batterie', linestyle='--', alpha=0.9)

        # Consumption line on top
        ax.plot(timestamps, consumption, color='#ffffff', linewidth=2,
                label='Verbrauch', linestyle='-')

        # Styling
        ax.set_ylabel('Leistung (kW)', color=self._styles.text_primary, fontsize=12)
        ax.set_title('Energiequellen', color=self._styles.text_primary, fontweight='bold', fontsize=14)

        # Format x-axis
        ax.tick_params(colors=self._styles.text_secondary)

        # Time format based on period
        import matplotlib.dates as mdates
        if self.period == 'today' or len(timestamps) < 500:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
            ax.xaxis.set_major_locator(mdates.DayLocator())

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Legend
        ax.legend(loc='upper right', facecolor=self._styles.background_card,
                  edgecolor=self._styles.border, fontsize=10)

        # Grid
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Add statistics box
        self._add_stats_box(ax)

    def _render_battery_soc(self, ax: Any) -> None:
        """Render battery SOC timeline. @zara"""
        import matplotlib.pyplot as plt

        if not self.data:
            ax.axis('off')
            return

        timestamps = []
        soc_values = []

        for point in self.data:
            ts = point.get('timestamp', '')
            soc = point.get('battery_soc')

            if ts and soc is not None:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    timestamps.append(dt)
                    soc_values.append(soc)
                except (ValueError, TypeError):
                    continue

        if not timestamps:
            ax.text(0.5, 0.5, 'Keine SOC-Daten', ha='center', va='center',
                    color=self._styles.text_muted, fontsize=10)
            ax.set_facecolor(self._styles.background)
            return

        # Color gradient based on SOC level
        battery_green = '#4DD0E1'  # Cyan color for battery
        ax.fill_between(timestamps, 0, soc_values,
                        color=battery_green, alpha=0.6)
        ax.plot(timestamps, soc_values, color=battery_green,
                linewidth=1.5)

        ax.set_ylabel('SOC (%)', color=self._styles.text_primary, fontsize=10)
        ax.set_xlabel('Zeit', color=self._styles.text_primary, fontsize=10)
        ax.set_ylim(0, 100)

        ax.tick_params(colors=self._styles.text_secondary, labelsize=8)
        ax.grid(True, alpha=0.2, color=self._styles.grid)
        ax.set_facecolor(self._styles.background)

        # Format x-axis
        import matplotlib.dates as mdates
        if self.period == 'today' or len(timestamps) < 500:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _add_stats_box(self, ax: Any) -> None:
        """Add statistics summary box. @zara"""
        if not self.stats:
            return

        # Calculate stats from data if not provided
        # Support both old (snake_case) and new (camelCase) keys
        solar_total = self.stats.get('solarTotal', self.stats.get('solar_total', 0)) or 0
        solar_to_house = self.stats.get('solarToHouse', self.stats.get('solar_to_house', 0)) or 0
        solar_to_battery = self.stats.get('solarToBattery', self.stats.get('solar_to_battery', 0)) or 0
        battery_total = self.stats.get('batteryTotal', self.stats.get('battery_total', 0)) or 0
        grid_total = self.stats.get('gridTotal', self.stats.get('grid_total', 0)) or 0
        consumption_total = self.stats.get('consumptionTotal', self.stats.get('consumption_total', 0)) or 0
        autarky = self.stats.get('autarky', 0) or 0

        # Calculate from data if stats empty
        if not solar_total and self.data:
            solar_to_house_vals = [p.get('solar_to_house', 0) or 0 for p in self.data]
            solar_to_battery_vals = [p.get('solar_to_battery', 0) or 0 for p in self.data]
            battery_vals = [p.get('battery_to_house', 0) or 0 for p in self.data]
            grid_vals = [p.get('grid_to_house', 0) or 0 for p in self.data]
            consumption_vals = [p.get('home_consumption', 0) or 0 for p in self.data]

            # Convert W to kWh: W * intervalHours / 1000
            # 5-min intervals = 5/60 hours per interval
            interval_hours = 5 / 60
            solar_to_house = (sum(solar_to_house_vals) * interval_hours) / 1000
            solar_to_battery = (sum(solar_to_battery_vals) * interval_hours) / 1000
            solar_total = solar_to_house + solar_to_battery
            battery_total = (sum(battery_vals) * interval_hours) / 1000
            grid_total = (sum(grid_vals) * interval_hours) / 1000
            consumption_total = (sum(consumption_vals) * interval_hours) / 1000

            if consumption_total > 0:
                autarky = ((solar_to_house + battery_total) / consumption_total) * 100
                autarky = min(100, autarky)

        stats_text = (
            f"Solar gesamt: {solar_total:.2f} kWh\n"
            f"  → Haus: {solar_to_house:.2f} kWh\n"
            f"  → Batterie: {solar_to_battery:.2f} kWh\n"
            f"Batterie → Haus: {battery_total:.2f} kWh\n"
            f"Netz → Haus: {grid_total:.2f} kWh\n"
            f"Verbrauch: {consumption_total:.2f} kWh\n"
            f"Autarkie: {autarky:.1f}%"
        )

        props = dict(
            boxstyle='round,pad=0.5',
            facecolor=self._styles.background_card,
            edgecolor=self._styles.border,
            alpha=0.9
        )

        ax.text(
            0.02, 0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            color=self._styles.text_primary,
            ha='left',
            va='top',
            bbox=props,
            family='monospace'
        )

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
            "SFML Stats - Power Sources",
            fontsize=9,
            color=self._styles.text_muted,
            ha='left',
            va='bottom',
            transform=fig.transFigure,
            style='italic',
        )
