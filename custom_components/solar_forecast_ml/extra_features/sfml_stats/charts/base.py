# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Base chart class for SFML Stats."""
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

# Shared executor for matplotlib operations
_MATPLOTLIB_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="matplotlib")


class BaseChart(ABC):
    """Abstrakte Basisklasse für alle Charts."""

    def __init__(
        self,
        validator: "DataValidator",
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        """Initialisiere das Chart.

        Args:
            validator: DataValidator Instanz
            figsize: Größe der Figure (Breite, Höhe) in Zoll
        """
        self._validator = validator
        self._figsize = figsize
        self._styles = ChartStyles()
        self._fig: "Figure | None" = None
        # Note: apply_dark_theme() is now called in executor when generating charts

    @property
    def styles(self) -> ChartStyles:
        """Gibt die Style-Konfiguration zurück."""
        return self._styles

    @property
    def export_path(self) -> Path:
        """Gibt den Export-Pfad für Charts zurück."""
        return self._validator.get_export_path("charts")

    @abstractmethod
    async def generate(self, **kwargs: Any) -> "Figure":
        """Generiert das Chart.

        WICHTIG: Subklassen sollten das Chart-Rendering in einem Executor ausführen,
        um den Event-Loop nicht zu blockieren. Nutze _run_in_executor() dafür.

        Args:
            **kwargs: Chart-spezifische Parameter

        Returns:
            Matplotlib Figure
        """

    async def _run_in_executor(self, func, *args, **kwargs):
        """Führt eine Funktion im Executor aus.

        Args:
            func: Die auszuführende Funktion
            *args: Positionale Argumente
            **kwargs: Keyword Argumente

        Returns:
            Das Ergebnis der Funktion
        """
        import functools
        loop = asyncio.get_running_loop()
        if kwargs:
            func = functools.partial(func, **kwargs)
        return await loop.run_in_executor(_MATPLOTLIB_EXECUTOR, func, *args)

    @abstractmethod
    def get_filename(self, **kwargs: Any) -> str:
        """Gibt den Dateinamen für das Chart zurück.

        Args:
            **kwargs: Parameter für den Dateinamen

        Returns:
            Dateiname (ohne Pfad)
        """

    async def save(self, filename: str | None = None, **kwargs: Any) -> Path:
        """Speichert das Chart als PNG.

        Args:
            filename: Optionaler Dateiname (sonst aus get_filename)
            **kwargs: Parameter für generate() und get_filename()

        Returns:
            Pfad zur gespeicherten Datei
        """
        # Chart generieren falls noch nicht geschehen
        if self._fig is None:
            self._fig = await self.generate(**kwargs)

        # Dateiname bestimmen
        if filename is None:
            filename = self.get_filename(**kwargs)

        # Vollständigen Pfad erstellen
        file_path = self.export_path / filename

        # Speichern im Executor
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
        """Erstellt eine neue Figure mit Subplots.

        WICHTIG: Diese Methode muss in einem Executor aufgerufen werden.

        Args:
            nrows: Anzahl Zeilen
            ncols: Anzahl Spalten
            figsize: Optionale Größe (sonst self._figsize)
            **kwargs: Weitere Parameter für subplots()

        Returns:
            Tuple aus (Figure, Axes)
        """
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
        """Fügt Titel und optionalen Untertitel hinzu.

        Args:
            ax: Matplotlib Axes
            title: Haupttitel
            subtitle: Optionaler Untertitel
        """
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
        """Fügt einen Footer mit Zeitstempel hinzu.

        Args:
            fig: Matplotlib Figure
            text: Optionaler zusätzlicher Text
        """
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

        # SFML Stats Branding
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
        """Fügt eine KPI-Box zum Chart hinzu.

        Args:
            ax: Matplotlib Axes
            kpis: Dictionary mit KPI-Namen und Werten
            position: Position ("right", "left", "top", "bottom")
        """
        # Text für die Box erstellen
        lines = []
        for name, value in kpis.items():
            if isinstance(value, float):
                value = f"{value:.1f}"
            lines.append(f"{name}: {value}")

        text = "\n".join(lines)

        # Position bestimmen
        positions = {
            "right": (0.98, 0.98, "right", "top"),
            "left": (0.02, 0.98, "left", "top"),
            "top": (0.5, 0.98, "center", "top"),
            "bottom": (0.5, 0.02, "center", "bottom"),
        }
        x, y, ha, va = positions.get(position, positions["right"])

        # Box erstellen
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
        """Erstellt Legende-Patches.

        WICHTIG: Diese Methode muss in einem Executor aufgerufen werden.

        Args:
            labels_colors: Dictionary mit Labels und Farben

        Returns:
            Liste von Matplotlib Patches
        """
        import matplotlib.patches as mpatches
        return [
            mpatches.Patch(color=color, label=label)
            for label, color in labels_colors.items()
        ]

    def _format_kwh(self, value: float) -> str:
        """Formatiert einen kWh-Wert.

        Args:
            value: Wert in kWh

        Returns:
            Formatierter String
        """
        if value >= 1000:
            return f"{value / 1000:.1f} MWh"
        elif value >= 1:
            return f"{value:.2f} kWh"
        else:
            return f"{value * 1000:.0f} Wh"

    def _format_price(self, value: float) -> str:
        """Formatiert einen Preiswert.

        Args:
            value: Wert in ct/kWh

        Returns:
            Formatierter String
        """
        return f"{value:.2f} ct/kWh"

    def _format_percent(self, value: float) -> str:
        """Formatiert einen Prozentwert.

        Args:
            value: Wert in Prozent

        Returns:
            Formatierter String
        """
        return f"{value:.1f}%"
