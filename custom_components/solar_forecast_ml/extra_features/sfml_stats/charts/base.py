# ******************************************************************************
# @copyright (C) 2026 Zara-Toorox - Solar Forecast Stats x86 DB-Version part of Solar Forecast ML DB
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Base chart class for SFML Stats. @zara"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .styles import ChartStyles
from ..const import CHART_DPI

if TYPE_CHECKING:
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    from ..storage import DataValidator

_LOGGER = logging.getLogger(__name__)

_MATPLOTLIB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="matplotlib")


class BaseChart(ABC):
    """Abstract base class for all charts. @zara"""

    def __init__(
        self,
        validator: "DataValidator",
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        """Initialize the chart. @zara"""
        self._validator = validator
        self._figsize = figsize
        self._styles = ChartStyles()
        self._fig: "Figure | None" = None

    @property
    def styles(self) -> ChartStyles:
        """Return the style configuration. @zara"""
        return self._styles

    @property
    def export_path(self) -> Path:
        """Return the export path for charts. @zara"""
        return self._validator.get_export_path("charts")

    @abstractmethod
    async def generate(self, **kwargs: Any) -> "Figure":
        """Generate the chart. @zara"""

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run a function in the executor. @zara"""
        import functools
        loop = asyncio.get_running_loop()
        if kwargs:
            func = functools.partial(func, **kwargs)
        return await loop.run_in_executor(_MATPLOTLIB_EXECUTOR, func, *args)

    @abstractmethod
    def get_filename(self, **kwargs: Any) -> str:
        """Return the filename for the chart. @zara"""

    async def save(self, filename: str | None = None, **kwargs: Any) -> Path:
        """Save the chart as PNG. @zara"""
        if self._fig is None:
            self._fig = await self.generate(**kwargs)

        if filename is None:
            filename = self.get_filename(**kwargs)

        file_path = self.export_path / filename

        def _save_sync():
            import matplotlib.pyplot as plt
            self._fig.savefig(
                file_path,
                dpi=CHART_DPI,
                bbox_inches="tight",
                facecolor=self._styles.background,
                edgecolor="none",
            )
            plt.close(self._fig)

        await self._run_in_executor(_save_sync)

        _LOGGER.info("Chart gespeichert: %s", file_path)
        self._fig = None

        return file_path

    def _create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> tuple["Figure", Any]:
        """Create a new figure with subplots. @zara"""
        import matplotlib.pyplot as plt
        size = figsize or self._figsize
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=size,
            facecolor=self._styles.background,
            **kwargs,
        )
        return fig, axes

    def _add_title(
        self,
        ax: Any,
        title: str,
        subtitle: str | None = None,
    ) -> None:
        """Add title and optional subtitle. @zara"""
        ax.set_title(
            title,
            fontsize=self._styles.title_size,
            fontweight="bold",
            color=self._styles.text_primary,
            pad=20,
        )

        if subtitle:
            ax.text(
                0.5, 1.02,
                subtitle,
                transform=ax.transAxes,
                fontsize=self._styles.subtitle_size,
                color=self._styles.text_secondary,
                ha="center",
                va="bottom",
            )

    def _add_footer(
        self,
        fig: "Figure",
        text: str | None = None,
    ) -> None:
        """Add footer with timestamp. @zara"""
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
        footer_text = f"Generiert: {timestamp}"

        if text:
            footer_text = f"{text} | {footer_text}"

        fig.text(
            0.99, 0.01,
            footer_text,
            fontsize=8,
            color=self._styles.text_muted,
            ha="right",
            va="bottom",
            transform=fig.transFigure,
        )

        fig.text(
            0.01, 0.01,
            "SFML Stats",
            fontsize=8,
            color=self._styles.text_muted,
            ha="left",
            va="bottom",
            transform=fig.transFigure,
            style="italic",
        )

    def _add_kpi_box(
        self,
        ax: Any,
        kpis: dict[str, str | float],
        position: str = "right",
    ) -> None:
        """Add a KPI box to the chart. @zara"""
        lines = []
        for name, value in kpis.items():
            if isinstance(value, float):
                value = f"{value:.1f}"
            lines.append(f"{name}: {value}")

        text = "\n".join(lines)

        positions = {
            "right": (0.98, 0.98, "right", "top"),
            "left": (0.02, 0.98, "left", "top"),
            "top": (0.5, 0.98, "center", "top"),
            "bottom": (0.5, 0.02, "center", "bottom"),
        }
        x, y, ha, va = positions.get(position, positions["right"])

        props = dict(
            boxstyle="round,pad=0.5",
            facecolor=self._styles.background_card,
            edgecolor=self._styles.border,
            alpha=0.9,
        )

        ax.text(
            x, y,
            text,
            transform=ax.transAxes,
            fontsize=self._styles.label_size,
            color=self._styles.text_primary,
            ha=ha,
            va=va,
            bbox=props,
            family="monospace",
        )

    def _create_legend_patches(
        self,
        labels_colors: dict[str, str],
    ) -> list["mpatches.Patch"]:
        """Create legend patches. @zara"""
        import matplotlib.patches as mpatches
        return [
            mpatches.Patch(color=color, label=label)
            for label, color in labels_colors.items()
        ]

    def _format_kwh(self, value: float) -> str:
        """Format a kWh value. @zara"""
        if value >= 1000:
            return f"{value / 1000:.1f} MWh"
        elif value >= 1:
            return f"{value:.2f} kWh"
        else:
            return f"{value * 1000:.0f} Wh"

    def _format_price(self, value: float) -> str:
        """Format a price value. @zara"""
        return f"{value:.2f} ct/kWh"

    def _format_percent(self, value: float) -> str:
        """Format a percentage value. @zara"""
        return f"{value:.1f}%"
