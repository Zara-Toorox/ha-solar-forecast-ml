# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Weekly report chart for SFML Stats - Modern Redesign."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from .base import BaseChart
from .styles import (
    ChartStyles,
    WEEKDAY_NAMES_DE,
    MONTH_NAMES_DE,
    COLOR_PALETTE_COMPARISON,
    add_glow_effect,
    draw_rounded_bar,
    draw_glass_box,
    create_gradient_image,
)
from ..const import (
    CHART_SIZE_WEEKLY,
    CHART_DPI,
    WEEKLY_REPORT_PATTERN,
    SFML_STATS_WEEKLY,
)
from ..readers import SolarDataReader, PriceDataReader

if TYPE_CHECKING:
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from ..storage import DataValidator

_LOGGER = logging.getLogger(__name__)


class WeeklyReportChart(BaseChart):
    """Generiert den wöchentlichen Report als Multi-Panel Chart."""

    def __init__(self, validator: DataValidator) -> None:
        """Initialisiere den WeeklyReportChart.

        Args:
            validator: DataValidator Instanz
        """
        super().__init__(validator, figsize=CHART_SIZE_WEEKLY)
        self._solar_reader = SolarDataReader(validator.config_path)
        self._price_reader = PriceDataReader(validator.config_path)

    @property
    def export_path(self) -> Path:
        """Gibt den Export-Pfad für Wochenberichte zurück."""
        return self._validator.get_export_path(SFML_STATS_WEEKLY)

    def get_filename(self, year: int = None, week: int = None, **kwargs) -> str:
        """Gibt den Dateinamen für den Wochenbericht zurück.

        Args:
            year: Jahr
            week: Kalenderwoche

        Returns:
            Dateiname
        """
        if year is None or week is None:
            today = date.today()
            year, week, _ = today.isocalendar()

        return WEEKLY_REPORT_PATTERN.format(week=week, year=year)

    async def generate(
        self,
        year: int = None,
        week: int = None,
        **kwargs,
    ) -> "Figure":
        """Generiert den kompletten Wochenbericht.

        Args:
            year: Jahr (default: aktuelles Jahr)
            week: Kalenderwoche (default: aktuelle Woche)

        Returns:
            Matplotlib Figure
        """
        # Defaults setzen
        if year is None or week is None:
            today = date.today()
            iso = today.isocalendar()
            year = year or iso[0]
            week = week or iso[1]

        _LOGGER.info("Generiere Wochenbericht für KW %d/%d", week, year)

        # Daten laden (async, außerhalb des Executors)
        solar_stats = await self._solar_reader.async_get_weekly_stats(year, week)
        price_stats = await self._price_reader.async_get_weekly_stats(year, week)
        hourly_predictions = await self._solar_reader.async_get_hourly_predictions()

        # Wochendaten ermitteln
        week_start = self._get_week_start(year, week)
        week_end = week_start + timedelta(days=6)

        # Chart im Executor generieren
        fig = await self._run_in_executor(
            self._generate_sync,
            year, week, week_start, week_end,
            solar_stats, price_stats, hourly_predictions
        )

        self._fig = fig
        return fig

    def _generate_sync(
        self,
        year: int,
        week: int,
        week_start: date,
        week_end: date,
        solar_stats: dict,
        price_stats: dict,
        hourly_predictions: list,
    ) -> "Figure":
        """Synchrones Chart-Rendering - läuft im Executor. Modern Redesign."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import FancyBboxPatch, Circle
        from .styles import apply_dark_theme

        apply_dark_theme()

        # Figure mit GridSpec erstellen (größer für mehr Details)
        fig = plt.figure(figsize=(18, 22), facecolor=self.styles.background)

        # GridSpec Layout: 5 Zeilen für bessere Aufteilung
        # Row 0: Header mit Gradient
        # Row 1: KPI Cards
        # Row 2: Produktion + Radiales Gauge
        # Row 3: Heatmaps (Preis + Genauigkeit)
        # Row 4: Korrelation + Footer
        gs = gridspec.GridSpec(
            5, 2,
            figure=fig,
            height_ratios=[0.6, 0.5, 1.1, 1.0, 1.3],
            width_ratios=[1.2, 0.8],
            hspace=0.3,
            wspace=0.2,
            left=0.06,
            right=0.94,
            top=0.96,
            bottom=0.04,
        )

        # Header (ganze Breite) - mit Gradient
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_modern_header(ax_header, year, week, week_start, week_end)

        # KPI Cards (ganze Breite)
        ax_kpi = fig.add_subplot(gs[1, :])
        self._draw_kpi_cards(ax_kpi, solar_stats, price_stats)

        # Chart 1: Produktion vs. Vorhersage (links) - mit abgerundeten Balken
        ax_production = fig.add_subplot(gs[2, 0])
        self._draw_modern_production_chart(ax_production, solar_stats)

        # Chart 2: Radiales Gauge für ML-Anteil + Wochenübersicht (rechts)
        ax_gauge = fig.add_subplot(gs[2, 1])
        self._draw_radial_gauge(ax_gauge, solar_stats)

        # Chart 3: Preis-Heatmap (links) - mit verbessertem Styling
        ax_price = fig.add_subplot(gs[3, 0])
        self._draw_modern_price_heatmap(ax_price, price_stats)

        # Chart 4: Genauigkeit Heatmap (rechts) - modernisiert
        ax_accuracy = fig.add_subplot(gs[3, 1])
        self._draw_modern_accuracy_heatmap(ax_accuracy, solar_stats)

        # Chart 5: Solar + Preis Korrelation (ganze Breite unten)
        ax_correlation = fig.add_subplot(gs[4, :])
        self._draw_modern_correlation(
            ax_correlation, solar_stats, price_stats,
            year, week, hourly_predictions
        )

        # Moderner Footer
        self._add_modern_footer(fig, year, week)

        return fig

    def _get_week_start(self, year: int, week: int) -> date:
        """Berechnet den Montag der angegebenen Kalenderwoche."""
        jan4 = date(year, 1, 4)
        start_of_week1 = jan4 - timedelta(days=jan4.weekday())
        return start_of_week1 + timedelta(weeks=week - 1)

    def _draw_modern_header(
        self,
        ax: "Axes",
        year: int,
        week: int,
        week_start: date,
        week_end: date,
    ) -> None:
        """Zeichnet den modernen Header mit Gradient-Effekt."""
        from matplotlib.patches import FancyBboxPatch, Rectangle
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.colors as mcolors

        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Gradient-Hintergrund für Header
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack([gradient] * 50)

        # Solar-Gradient (Gold -> Orange -> leichtes Rot)
        colors = [self.styles.solar_gold, self.styles.solar_orange, "#ff6b35"]
        cmap = LinearSegmentedColormap.from_list("solar_header", colors)

        ax.imshow(
            gradient,
            aspect="auto",
            cmap=cmap,
            extent=[0, 1, 0, 1],
            alpha=0.15,
        )

        # Sonnen-Symbol (Unicode)
        ax.text(
            0.05, 0.5,
            "☀",
            transform=ax.transAxes,
            fontsize=48,
            color=self.styles.solar_yellow,
            ha="left",
            va="center",
            alpha=0.9,
        )

        # Titel
        month_name = MONTH_NAMES_DE[week_start.month - 1]
        title = f"SFML Stats · Wochenbericht"
        ax.text(
            0.5, 0.7,
            title,
            transform=ax.transAxes,
            fontsize=26,
            fontweight="bold",
            color=self.styles.text_primary,
            ha="center",
            va="center",
            family=self.styles.font_family,
        )

        # KW und Datum
        kw_text = f"KW {week}"
        ax.text(
            0.5, 0.38,
            kw_text,
            transform=ax.transAxes,
            fontsize=36,
            fontweight="bold",
            color=self.styles.neon_cyan,
            ha="center",
            va="center",
        )

        # Datumsbereich
        date_range = f"{week_start.strftime('%d.%m.')} – {week_end.strftime('%d.%m.%Y')} · {month_name}"
        ax.text(
            0.5, 0.12,
            date_range,
            transform=ax.transAxes,
            fontsize=14,
            color=self.styles.text_secondary,
            ha="center",
            va="center",
        )

        # Dezente Linie unter dem Header
        ax.axhline(y=0.02, xmin=0.1, xmax=0.9, color=self.styles.border, linewidth=1, alpha=0.5)

    def _draw_kpi_cards(
        self,
        ax: "Axes",
        solar_stats: dict,
        price_stats: dict,
    ) -> None:
        """Zeichnet die KPI-Karten im modernen Glassmorphism-Stil."""
        from matplotlib.patches import FancyBboxPatch

        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # KPI-Daten sammeln
        kpis = []

        if solar_stats.get("data_available"):
            kpis.extend([
                {
                    "value": f"{solar_stats.get('total_actual_kwh', 0):.1f}",
                    "unit": "kWh",
                    "label": "Produktion",
                    "color": self.styles.solar_yellow,
                    "icon": "⚡",
                },
                {
                    "value": f"{solar_stats.get('average_accuracy_percent', 0):.0f}",
                    "unit": "%",
                    "label": "Genauigkeit",
                    "color": self._get_accuracy_color(solar_stats.get('average_accuracy_percent', 0)),
                    "icon": "🎯",
                },
                {
                    "value": f"{solar_stats.get('avg_ml_contribution_percent', 0):.0f}",
                    "unit": "%",
                    "label": "ML-Anteil",
                    "color": self.styles.ml_purple,
                    "icon": "🤖",
                },
            ])
        else:
            kpis.append({
                "value": "–",
                "unit": "",
                "label": "Produktion",
                "color": self.styles.text_muted,
                "icon": "⚡",
            })

        if price_stats.get("data_available"):
            kpis.extend([
                {
                    "value": f"{price_stats.get('average_price', 0):.1f}",
                    "unit": "ct",
                    "label": "Ø Preis",
                    "color": self.styles.solar_orange,
                    "icon": "💰",
                },
                {
                    "value": f"{price_stats.get('min_price', 0):.1f}",
                    "unit": "ct",
                    "label": "Min",
                    "color": self.styles.price_green,
                    "icon": "📉",
                },
                {
                    "value": f"{price_stats.get('max_price', 0):.1f}",
                    "unit": "ct",
                    "label": "Max",
                    "color": self.styles.price_red,
                    "icon": "📈",
                },
            ])

        # KPI-Karten zeichnen
        num_kpis = len(kpis)
        card_width = 0.13
        spacing = 0.02
        total_width = num_kpis * card_width + (num_kpis - 1) * spacing
        start_x = (1 - total_width) / 2

        for i, kpi in enumerate(kpis):
            x = start_x + i * (card_width + spacing) + card_width / 2

            # Karten-Hintergrund mit Glow
            card_bg = FancyBboxPatch(
                (x - card_width/2, 0.15),
                card_width,
                0.7,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                facecolor=self.styles.background_card,
                edgecolor=kpi["color"],
                linewidth=2,
                alpha=0.9,
                transform=ax.transAxes,
            )
            ax.add_patch(card_bg)

            # Icon
            ax.text(
                x, 0.72,
                kpi["icon"],
                transform=ax.transAxes,
                fontsize=18,
                ha="center",
                va="center",
            )

            # Wert + Einheit
            ax.text(
                x, 0.48,
                kpi["value"],
                transform=ax.transAxes,
                fontsize=22,
                fontweight="bold",
                color=kpi["color"],
                ha="center",
                va="center",
            )
            ax.text(
                x, 0.32,
                kpi["unit"],
                transform=ax.transAxes,
                fontsize=12,
                color=self.styles.text_secondary,
                ha="center",
                va="center",
            )

            # Label
            ax.text(
                x, 0.18,
                kpi["label"],
                transform=ax.transAxes,
                fontsize=10,
                color=self.styles.text_secondary,
                ha="center",
                va="center",
            )

    def _get_accuracy_color(self, accuracy: float) -> str:
        """Gibt die passende Farbe für die Genauigkeit zurück."""
        if accuracy >= 80:
            return self.styles.accuracy_good
        elif accuracy >= 50:
            return self.styles.accuracy_medium
        else:
            return self.styles.accuracy_bad

    def _draw_header(
        self,
        ax: "Axes",
        year: int,
        week: int,
        week_start: date,
        week_end: date,
        solar_stats: dict,
        price_stats: dict,
    ) -> None:
        """Zeichnet den Header mit Titel und KPIs."""
        ax.axis("off")

        # Titel
        month_name = MONTH_NAMES_DE[week_start.month - 1]
        title = f"SFML Stats - Wochenbericht KW {week}"
        subtitle = f"{week_start.strftime('%d.%m.')} - {week_end.strftime('%d.%m.%Y')} ({month_name})"

        ax.text(
            0.5, 0.85,
            title,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            color=self.styles.text_primary,
            ha="center",
            va="top",
        )

        ax.text(
            0.5, 0.65,
            subtitle,
            transform=ax.transAxes,
            fontsize=14,
            color=self.styles.text_secondary,
            ha="center",
            va="top",
        )

        # KPI-Boxen
        kpi_y = 0.25
        box_props = dict(
            boxstyle="round,pad=0.4",
            facecolor=self.styles.background_card,
            edgecolor=self.styles.border,
            alpha=0.9,
        )

        # Solar KPIs
        if solar_stats.get("data_available"):
            solar_kpis = [
                (f"{solar_stats.get('total_actual_kwh', 0):.1f} kWh", "Produktion", self.styles.solar_yellow),
                (f"{solar_stats.get('average_accuracy_percent', 0):.0f}%", "Genauigkeit", self.styles.accuracy_medium),
                (f"{solar_stats.get('avg_ml_contribution_percent', 0):.0f}%", "ML-Anteil", self.styles.ml_purple),
            ]
        else:
            solar_kpis = [("--", "Produktion", self.styles.text_muted)]

        # Preis KPIs
        if price_stats.get("data_available"):
            price_kpis = [
                (f"{price_stats.get('average_price', 0):.1f} ct", "Ø Preis", self.styles.solar_orange),
                (f"{price_stats.get('min_price', 0):.1f} ct", "Min", self.styles.price_green),
                (f"{price_stats.get('max_price', 0):.1f} ct", "Max", self.styles.price_red),
            ]
        else:
            price_kpis = [("--", "Ø Preis", self.styles.text_muted)]

        all_kpis = solar_kpis + price_kpis
        positions = np.linspace(0.08, 0.92, len(all_kpis))

        for (value, label, color), x_pos in zip(all_kpis, positions):
            ax.text(
                x_pos, kpi_y + 0.12,
                value,
                transform=ax.transAxes,
                fontsize=16,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                bbox=box_props,
            )
            ax.text(
                x_pos, kpi_y - 0.08,
                label,
                transform=ax.transAxes,
                fontsize=10,
                color=self.styles.text_secondary,
                ha="center",
                va="center",
            )

    def _draw_production_chart(self, ax: "Axes", solar_stats: dict) -> None:
        """Zeichnet das Produktions-Balkendiagramm (Vorhersage vs. Actual)."""
        ax.set_facecolor(self.styles.background_light)

        if not solar_stats.get("data_available") or not solar_stats.get("daily_summaries"):
            ax.text(
                0.5, 0.5,
                "Keine Solardaten verfügbar",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=12,
            )
            ax.set_title("Produktion vs. Vorhersage", fontsize=12, color=self.styles.text_primary)
            return

        summaries = sorted(solar_stats["daily_summaries"], key=lambda x: x.date)

        days = [WEEKDAY_NAMES_DE[s.day_of_week] for s in summaries]
        predicted = [s.predicted_total_kwh for s in summaries]
        actual = [s.actual_total_kwh for s in summaries]

        x = np.arange(len(days))
        width = 0.35

        bars_pred = ax.bar(
            x - width/2, predicted, width,
            label="Vorhersage",
            color=self.styles.predicted,
            alpha=0.8,
            edgecolor=self.styles.border,
        )
        bars_actual = ax.bar(
            x + width/2, actual, width,
            label="Tatsächlich",
            color=self.styles.actual,
            alpha=0.8,
            edgecolor=self.styles.border,
        )

        # Werte über Balken
        for bar, val in zip(bars_actual, actual):
            if val > 0:
                ax.annotate(
                    f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8,
                    color=self.styles.text_secondary,
                )

        ax.set_xlabel("Wochentag", fontsize=10)
        ax.set_ylabel("Energie (kWh)", fontsize=10)
        ax.set_title("Produktion vs. Vorhersage", fontsize=12, fontweight="bold", color=self.styles.text_primary)
        ax.set_xticks(x)
        ax.set_xticklabels(days)
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(bottom=0)

    def _draw_ml_contribution_chart(self, ax: "Axes", solar_stats: dict) -> None:
        """Zeichnet das ML vs. Rule-Based Diagramm."""
        ax.set_facecolor(self.styles.background_light)

        if not solar_stats.get("data_available"):
            ax.text(
                0.5, 0.5,
                "Keine ML-Daten verfügbar",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=12,
            )
            ax.set_title("ML vs. Rule-Based Anteil", fontsize=12, color=self.styles.text_primary)
            return

        # Durchschnittswerte für die Woche
        ml_percent = solar_stats.get("avg_ml_contribution_percent", 0)
        rb_percent = 100 - ml_percent

        # Donut Chart
        sizes = [ml_percent, rb_percent]
        colors = [self.styles.ml_purple, self.styles.rule_based_blue]
        labels = [f"ML\n{ml_percent:.0f}%", f"Rule-Based\n{rb_percent:.0f}%"]
        explode = (0.02, 0.02)

        wedges, texts = ax.pie(
            sizes,
            colors=colors,
            explode=explode,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor=self.styles.background),
        )

        # Labels außerhalb
        for i, (wedge, label) in enumerate(zip(wedges, labels)):
            ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
            x = np.cos(np.deg2rad(ang))
            y = np.sin(np.deg2rad(ang))
            ax.annotate(
                label,
                xy=(x * 0.75, y * 0.75),
                ha="center", va="center",
                fontsize=11,
                fontweight="bold",
                color=colors[i],
            )

        # Zentrum-Text
        ax.text(
            0, 0,
            "Vorhersage-\nMethode",
            ha="center", va="center",
            fontsize=10,
            color=self.styles.text_secondary,
        )

        ax.set_title("ML vs. Rule-Based Anteil", fontsize=12, fontweight="bold", color=self.styles.text_primary)

    def _draw_price_heatmap(self, ax: "Axes", price_stats: dict) -> None:
        """Zeichnet die Preis-Heatmap (Stunde vs. Tag)."""
        ax.set_facecolor(self.styles.background_light)

        if not price_stats.get("data_available") or not price_stats.get("hourly_prices"):
            ax.text(
                0.5, 0.5,
                "Keine Preisdaten verfügbar",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=12,
            )
            ax.set_title("Strompreise (ct/kWh)", fontsize=12, color=self.styles.text_primary)
            return

        # Daten in Matrix umwandeln (7 Tage x 24 Stunden)
        prices = price_stats["hourly_prices"]

        # Gruppieren nach Tag
        days_data: dict[date, dict[int, float]] = {}
        for p in prices:
            d = p.date
            if d not in days_data:
                days_data[d] = {}
            days_data[d][p.hour] = p.price_net

        # Matrix erstellen
        sorted_days = sorted(days_data.keys())
        matrix = np.zeros((24, len(sorted_days)))
        matrix[:] = np.nan

        for col, day in enumerate(sorted_days):
            for hour, price in days_data[day].items():
                matrix[hour, col] = price

        # Heatmap
        import matplotlib.pyplot as plt
        from .styles import create_price_colormap
        cmap = create_price_colormap()
        im = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0,
            vmax=np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 30,
        )

        # Achsen
        ax.set_yticks(np.arange(0, 24, 3))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
        ax.set_xticks(np.arange(len(sorted_days)))
        day_labels = [WEEKDAY_NAMES_DE[d.weekday()] for d in sorted_days]
        ax.set_xticklabels(day_labels)

        ax.set_xlabel("Wochentag", fontsize=10)
        ax.set_ylabel("Uhrzeit", fontsize=10)
        ax.set_title("Strompreise (ct/kWh)", fontsize=12, fontweight="bold", color=self.styles.text_primary)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("ct/kWh", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    def _draw_accuracy_heatmap(self, ax: "Axes", solar_stats: dict) -> None:
        """Zeichnet die Genauigkeits-Heatmap."""
        ax.set_facecolor(self.styles.background_light)

        if not solar_stats.get("data_available") or not solar_stats.get("daily_summaries"):
            ax.text(
                0.5, 0.5,
                "Keine Genauigkeitsdaten verfügbar",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=12,
            )
            ax.set_title("Vorhersage-Genauigkeit", fontsize=12, color=self.styles.text_primary)
            return

        summaries = sorted(solar_stats["daily_summaries"], key=lambda x: x.date)

        # Zeitfenster-Daten extrahieren
        time_windows = ["Morgen\n7-10h", "Mittag\n11-14h", "Nachmittag\n15-17h"]
        matrix = np.zeros((3, len(summaries)))
        matrix[:] = np.nan

        for col, s in enumerate(summaries):
            if s.morning_accuracy is not None:
                # Cap accuracy at 150% for visualization
                matrix[0, col] = min(s.morning_accuracy, 150)
            if s.midday_accuracy is not None:
                matrix[1, col] = min(s.midday_accuracy, 150)
            if s.afternoon_accuracy is not None:
                matrix[2, col] = min(s.afternoon_accuracy, 150)

        # Heatmap (100% = perfekt, grün)
        import matplotlib.pyplot as plt
        from .styles import create_accuracy_colormap
        cmap = create_accuracy_colormap()
        im = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=150,
        )

        # Werte in Zellen anzeigen
        for i in range(3):
            for j in range(len(summaries)):
                val = matrix[i, j]
                if not np.isnan(val):
                    # Textfarbe basierend auf Hintergrund
                    text_color = "white" if val < 50 or val > 120 else "black"
                    ax.text(
                        j, i,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=8,
                        color=text_color,
                        fontweight="bold",
                    )

        # Achsen
        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(time_windows, fontsize=9)
        ax.set_xticks(np.arange(len(summaries)))
        day_labels = [WEEKDAY_NAMES_DE[s.day_of_week] for s in summaries]
        ax.set_xticklabels(day_labels)

        ax.set_xlabel("Wochentag", fontsize=10)
        ax.set_title("Vorhersage-Genauigkeit (%)", fontsize=12, fontweight="bold", color=self.styles.text_primary)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Genauigkeit %", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    def _draw_solar_price_correlation_sync(
        self,
        ax: "Axes",
        solar_stats: dict,
        price_stats: dict,
        year: int,
        week: int,
        hourly_predictions: list,
    ) -> None:
        """Zeichnet die Solar-Preis-Korrelation (synchrone Version für Executor)."""
        import matplotlib.pyplot as plt

        ax.set_facecolor(self.styles.background_light)

        has_solar = solar_stats.get("data_available") and solar_stats.get("daily_summaries")
        has_price = price_stats.get("data_available") and price_stats.get("hourly_prices")

        if not has_solar or not has_price:
            ax.text(
                0.5, 0.5,
                "Nicht genügend Daten für Korrelationsanalyse",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=12,
            )
            ax.set_title("Solar-Produktion & Strompreis Korrelation", fontsize=12, color=self.styles.text_primary)
            return

        # Nach Woche filtern
        week_predictions = [
            p for p in hourly_predictions
            if p.target_date.isocalendar()[0] == year
            and p.target_date.isocalendar()[1] == week
            and p.actual_kwh is not None
            and p.actual_kwh > 0
        ]

        if not week_predictions:
            ax.text(
                0.5, 0.5,
                "Keine stündlichen Produktionsdaten mit Preis-Überlappung",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=12,
            )
            ax.set_title("Solar-Produktion & Strompreis Korrelation", fontsize=12, color=self.styles.text_primary)
            return

        # Preise als Dict für schnellen Zugriff
        price_dict: dict[tuple[date, int], float] = {}
        for p in price_stats["hourly_prices"]:
            price_dict[(p.date, p.hour)] = p.price_net

        # Daten zusammenführen
        hours = []
        productions = []
        prices = []
        sizes = []

        for pred in week_predictions:
            key = (pred.target_date, pred.target_hour)
            if key in price_dict:
                hours.append(pred.target_hour)
                productions.append(pred.actual_kwh)
                prices.append(price_dict[key])
                # Größe basierend auf Produktion
                sizes.append(max(20, pred.actual_kwh * 200))

        if not hours:
            ax.text(
                0.5, 0.5,
                "Keine überlappenden Daten gefunden",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=12,
            )
            return

        # Scatter Plot
        scatter = ax.scatter(
            hours,
            prices,
            s=sizes,
            c=productions,
            cmap="YlOrRd",
            alpha=0.7,
            edgecolors=self.styles.border,
            linewidths=0.5,
        )

        # Durchschnittspreis-Linie
        avg_price = np.mean(prices)
        ax.axhline(
            y=avg_price,
            color=self.styles.solar_orange,
            linestyle="--",
            linewidth=2,
            label=f"Ø Preis: {avg_price:.1f} ct/kWh",
        )

        # Produktionsstunden markieren
        production_hours = sorted(set(hours))
        for h in production_hours:
            ax.axvspan(h - 0.5, h + 0.5, alpha=0.1, color=self.styles.solar_yellow)

        ax.set_xlabel("Stunde", fontsize=10)
        ax.set_ylabel("Strompreis (ct/kWh)", fontsize=10)
        ax.set_title(
            "Solar-Produktion & Strompreis (Punktgröße = Produktion kWh)",
            fontsize=12,
            fontweight="bold",
            color=self.styles.text_primary,
        )

        ax.set_xlim(5, 20)
        ax.set_xticks(range(6, 20, 2))
        ax.set_xticklabels([f"{h}:00" for h in range(6, 20, 2)])

        ax.legend(loc="upper right", fontsize=9)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Produktion (kWh)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # KPI Box
        total_production = sum(productions)
        weighted_avg_price = sum(p * prod for p, prod in zip(prices, productions)) / total_production if total_production > 0 else 0
        estimated_value = total_production * weighted_avg_price / 100  # in Euro

        kpi_text = (
            f"Σ Produktion: {total_production:.2f} kWh\n"
            f"Ø Einspeispreis: {weighted_avg_price:.1f} ct/kWh\n"
            f"Geschätzter Wert: {estimated_value:.2f} €"
        )

        props = dict(
            boxstyle="round,pad=0.5",
            facecolor=self.styles.background_card,
            edgecolor=self.styles.solar_yellow,
            alpha=0.95,
        )

        ax.text(
            0.02, 0.98,
            kpi_text,
            transform=ax.transAxes,
            fontsize=10,
            color=self.styles.text_primary,
            ha="left",
            va="top",
            bbox=props,
            family="monospace",
        )

    # =========================================================================
    # NEUE MODERNE CHART-METHODEN
    # =========================================================================

    def _draw_modern_production_chart(self, ax: "Axes", solar_stats: dict) -> None:
        """Zeichnet das modernisierte Produktions-Balkendiagramm mit Glow-Effekten."""
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.colors import LinearSegmentedColormap

        ax.set_facecolor(self.styles.background_light)

        if not solar_stats.get("data_available") or not solar_stats.get("daily_summaries"):
            ax.text(
                0.5, 0.5,
                "Keine Solardaten verfügbar",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=14,
            )
            ax.set_title("Produktion vs. Vorhersage", fontsize=14, color=self.styles.text_primary)
            return

        summaries = sorted(solar_stats["daily_summaries"], key=lambda x: x.date)

        days = [WEEKDAY_NAMES_DE[s.day_of_week] for s in summaries]
        predicted = [s.predicted_total_kwh for s in summaries]
        actual = [s.actual_total_kwh for s in summaries]

        x = np.arange(len(days))
        width = 0.35

        # Hintergrund-Gradient für den Chart-Bereich
        max_val = max(max(predicted) if predicted else 0, max(actual) if actual else 0) * 1.15

        # Balken mit Glow-Effekt für Vorhersage
        for i, (xi, val) in enumerate(zip(x, predicted)):
            if val > 0:
                # Glow (mehrere transparente Rechtecke)
                for alpha, expand in [(0.1, 0.08), (0.15, 0.04), (0.2, 0.02)]:
                    glow_patch = FancyBboxPatch(
                        (xi - width/2 - expand - width/2 - 0.02, 0),
                        width + expand * 2,
                        val,
                        boxstyle="round,pad=0,rounding_size=0.08",
                        facecolor=self.styles.predicted,
                        alpha=alpha,
                    )
                    ax.add_patch(glow_patch)

                # Hauptbalken
                bar_patch = FancyBboxPatch(
                    (xi - width/2 - width/2 - 0.02, 0),
                    width,
                    val,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    facecolor=self.styles.predicted,
                    edgecolor=self.styles.border,
                    linewidth=0.5,
                    alpha=0.85,
                )
                ax.add_patch(bar_patch)

        # Balken mit Glow-Effekt für Tatsächlich
        for i, (xi, val) in enumerate(zip(x, actual)):
            if val > 0:
                # Glow
                for alpha, expand in [(0.15, 0.08), (0.2, 0.04), (0.25, 0.02)]:
                    glow_patch = FancyBboxPatch(
                        (xi + width/2 - width/2 + 0.02 - expand, 0),
                        width + expand * 2,
                        val,
                        boxstyle="round,pad=0,rounding_size=0.08",
                        facecolor=self.styles.actual,
                        alpha=alpha,
                    )
                    ax.add_patch(glow_patch)

                # Hauptbalken
                bar_patch = FancyBboxPatch(
                    (xi + width/2 - width/2 + 0.02, 0),
                    width,
                    val,
                    boxstyle="round,pad=0,rounding_size=0.08",
                    facecolor=self.styles.actual,
                    edgecolor=self.styles.border,
                    linewidth=0.5,
                    alpha=0.9,
                )
                ax.add_patch(bar_patch)

                # Wert über dem Balken
                ax.annotate(
                    f"{val:.2f}",
                    xy=(xi + 0.02, val),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color=self.styles.neon_cyan,
                )

        ax.set_xlim(-0.6, len(days) - 0.4)
        ax.set_ylim(0, max_val)
        ax.set_xticks(x)
        ax.set_xticklabels(days, fontsize=11)
        ax.set_ylabel("Energie (kWh)", fontsize=11)

        # Titel mit Icon
        ax.set_title(
            "⚡ Produktion vs. Vorhersage",
            fontsize=14,
            fontweight="bold",
            color=self.styles.text_primary,
            pad=15,
        )

        # Moderne Legende
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.styles.predicted, alpha=0.85, label="Vorhersage"),
            Patch(facecolor=self.styles.actual, alpha=0.9, label="Tatsächlich"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=10,
            framealpha=0.9,
            edgecolor=self.styles.border,
        )

        # Dezente horizontale Linien
        ax.yaxis.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.xaxis.grid(False)

    def _draw_radial_gauge(self, ax: "Axes", solar_stats: dict) -> None:
        """Zeichnet ein radiales Gauge für ML-Anteil und Wochenübersicht."""
        from matplotlib.patches import Wedge, Circle, FancyBboxPatch
        import matplotlib.patheffects as path_effects

        ax.set_facecolor(self.styles.background_light)
        ax.axis("equal")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")

        if not solar_stats.get("data_available"):
            ax.text(
                0, 0,
                "Keine Daten",
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=14,
            )
            return

        # ML vs Rule-Based Werte
        ml_percent = solar_stats.get("avg_ml_contribution_percent", 0)
        rb_percent = 100 - ml_percent
        accuracy = solar_stats.get("average_accuracy_percent", 0)

        # Äußerer Ring: ML-Anteil
        # Hintergrund-Ring
        bg_ring = Wedge(
            (0, 0), 1.2, 0, 360,
            width=0.25,
            facecolor=self.styles.background_card,
            edgecolor=self.styles.border,
            linewidth=1,
        )
        ax.add_patch(bg_ring)

        # ML-Anteil (Lila)
        if ml_percent > 0:
            ml_angle = 360 * ml_percent / 100
            ml_wedge = Wedge(
                (0, 0), 1.2, 90, 90 - ml_angle,
                width=0.25,
                facecolor=self.styles.ml_purple,
                edgecolor="none",
                alpha=0.9,
            )
            ax.add_patch(ml_wedge)

        # Rule-Based-Anteil (Blau)
        if rb_percent > 0:
            rb_wedge = Wedge(
                (0, 0), 1.2, 90 - 360 * ml_percent / 100, 90 - 360,
                width=0.25,
                facecolor=self.styles.rule_based_blue,
                edgecolor="none",
                alpha=0.9,
            )
            ax.add_patch(rb_wedge)

        # Innerer Ring: Genauigkeit
        inner_bg = Wedge(
            (0, 0), 0.85, 0, 360,
            width=0.2,
            facecolor=self.styles.background_card,
            edgecolor=self.styles.border,
            linewidth=1,
        )
        ax.add_patch(inner_bg)

        # Genauigkeits-Ring
        acc_color = self._get_accuracy_color(accuracy)
        acc_angle = min(360, 360 * accuracy / 100)
        if accuracy > 0:
            acc_wedge = Wedge(
                (0, 0), 0.85, 90, 90 - acc_angle,
                width=0.2,
                facecolor=acc_color,
                edgecolor="none",
                alpha=0.9,
            )
            ax.add_patch(acc_wedge)

        # Innerer Kreis
        inner_circle = Circle(
            (0, 0), 0.55,
            facecolor=self.styles.background_light,
            edgecolor=self.styles.border,
            linewidth=2,
        )
        ax.add_patch(inner_circle)

        # Zentrale Werte
        ax.text(
            0, 0.15,
            f"{ml_percent:.0f}%",
            ha="center", va="center",
            fontsize=28,
            fontweight="bold",
            color=self.styles.ml_purple,
        )
        ax.text(
            0, -0.15,
            "ML-Anteil",
            ha="center", va="center",
            fontsize=11,
            color=self.styles.text_secondary,
        )

        # Labels außen
        ax.text(
            0, 1.45,
            "🤖 ML",
            ha="center", va="center",
            fontsize=11,
            fontweight="bold",
            color=self.styles.ml_purple,
        )
        ax.text(
            0, -1.45,
            "📐 Rule-Based",
            ha="center", va="center",
            fontsize=11,
            fontweight="bold",
            color=self.styles.rule_based_blue,
        )

        # Genauigkeit rechts
        ax.text(
            1.4, 0,
            f"🎯\n{accuracy:.0f}%",
            ha="center", va="center",
            fontsize=12,
            fontweight="bold",
            color=acc_color,
        )

    def _draw_modern_price_heatmap(self, ax: "Axes", price_stats: dict) -> None:
        """Zeichnet die modernisierte Preis-Heatmap."""
        ax.set_facecolor(self.styles.background_light)

        if not price_stats.get("data_available") or not price_stats.get("hourly_prices"):
            ax.text(
                0.5, 0.5,
                "Keine Preisdaten verfügbar",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=14,
            )
            ax.set_title("💰 Strompreise", fontsize=14, color=self.styles.text_primary)
            return

        prices = price_stats["hourly_prices"]

        # Gruppieren nach Tag
        days_data: dict[date, dict[int, float]] = {}
        for p in prices:
            d = p.date
            if d not in days_data:
                days_data[d] = {}
            days_data[d][p.hour] = p.price_net

        # Matrix erstellen
        sorted_days = sorted(days_data.keys())
        matrix = np.zeros((24, len(sorted_days)))
        matrix[:] = np.nan

        for col, day in enumerate(sorted_days):
            for hour, price in days_data[day].items():
                matrix[hour, col] = price

        # Moderne Colormap (Grün -> Gelb -> Orange -> Rot)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        colors = [
            self.styles.price_green,
            self.styles.price_yellow,
            self.styles.solar_orange,
            self.styles.price_red,
        ]
        cmap = LinearSegmentedColormap.from_list("modern_price", colors, N=256)

        im = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            interpolation="bilinear",  # Sanfterer Übergang
            vmin=np.nanmin(matrix) if not np.all(np.isnan(matrix)) else 0,
            vmax=np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 30,
        )

        # Achsen
        ax.set_yticks(np.arange(0, 24, 4))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 4)], fontsize=10)
        ax.set_xticks(np.arange(len(sorted_days)))
        day_labels = [WEEKDAY_NAMES_DE[d.weekday()] for d in sorted_days]
        ax.set_xticklabels(day_labels, fontsize=11)

        ax.set_ylabel("Uhrzeit", fontsize=11)
        ax.set_title(
            "💰 Strompreise (ct/kWh)",
            fontsize=14,
            fontweight="bold",
            color=self.styles.text_primary,
            pad=10,
        )

        # Moderne Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=20)
        cbar.set_label("ct/kWh", fontsize=10)
        cbar.ax.tick_params(labelsize=9)
        cbar.outline.set_edgecolor(self.styles.border)

    def _draw_modern_accuracy_heatmap(self, ax: "Axes", solar_stats: dict) -> None:
        """Zeichnet die modernisierte Genauigkeits-Heatmap."""
        ax.set_facecolor(self.styles.background_light)

        if not solar_stats.get("data_available") or not solar_stats.get("daily_summaries"):
            ax.text(
                0.5, 0.5,
                "Keine Daten verfügbar",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=14,
            )
            ax.set_title("🎯 Vorhersage-Genauigkeit", fontsize=14, color=self.styles.text_primary)
            return

        summaries = sorted(solar_stats["daily_summaries"], key=lambda x: x.date)

        time_windows = ["Morgen\n(7-10h)", "Mittag\n(11-14h)", "Nachm.\n(15-17h)"]
        matrix = np.zeros((3, len(summaries)))
        matrix[:] = np.nan

        for col, s in enumerate(summaries):
            if s.morning_accuracy is not None:
                matrix[0, col] = min(s.morning_accuracy, 150)
            if s.midday_accuracy is not None:
                matrix[1, col] = min(s.midday_accuracy, 150)
            if s.afternoon_accuracy is not None:
                matrix[2, col] = min(s.afternoon_accuracy, 150)

        # Moderne Colormap mit besserem Kontrast
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        # Divergierende Colormap: Rot (schlecht) -> Gelb (mittel) -> Grün (gut)
        colors = [
            self.styles.accuracy_bad,
            self.styles.accuracy_medium,
            self.styles.accuracy_good,
        ]
        cmap = LinearSegmentedColormap.from_list("modern_accuracy", colors, N=256)

        im = ax.imshow(
            matrix,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            vmin=0,
            vmax=150,
        )

        # Werte in Zellen mit modernem Styling
        for i in range(3):
            for j in range(len(summaries)):
                val = matrix[i, j]
                if not np.isnan(val):
                    # Dynamische Textfarbe
                    if val >= 80:
                        text_color = self.styles.background
                    elif val >= 50:
                        text_color = self.styles.background
                    else:
                        text_color = self.styles.text_primary

                    ax.text(
                        j, i,
                        f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=11,
                        color=text_color,
                        fontweight="bold",
                    )

        ax.set_yticks(np.arange(3))
        ax.set_yticklabels(time_windows, fontsize=10)
        ax.set_xticks(np.arange(len(summaries)))
        day_labels = [WEEKDAY_NAMES_DE[s.day_of_week] for s in summaries]
        ax.set_xticklabels(day_labels, fontsize=11)

        ax.set_title(
            "🎯 Vorhersage-Genauigkeit",
            fontsize=14,
            fontweight="bold",
            color=self.styles.text_primary,
            pad=10,
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.03, aspect=15)
        cbar.set_label("Genauigkeit %", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    def _draw_modern_correlation(
        self,
        ax: "Axes",
        solar_stats: dict,
        price_stats: dict,
        year: int,
        week: int,
        hourly_predictions: list,
    ) -> None:
        """Zeichnet die modernisierte Solar-Preis-Korrelation."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        from matplotlib.colors import LinearSegmentedColormap

        ax.set_facecolor(self.styles.background_light)

        has_solar = solar_stats.get("data_available") and solar_stats.get("daily_summaries")
        has_price = price_stats.get("data_available") and price_stats.get("hourly_prices")

        if not has_solar or not has_price:
            ax.text(
                0.5, 0.5,
                "Nicht genügend Daten für Analyse",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=14,
            )
            ax.set_title("☀️ Solar & Preis Analyse", fontsize=14, color=self.styles.text_primary)
            return

        # Daten filtern
        week_predictions = [
            p for p in hourly_predictions
            if p.target_date.isocalendar()[0] == year
            and p.target_date.isocalendar()[1] == week
            and p.actual_kwh is not None
            and p.actual_kwh > 0
        ]

        if not week_predictions:
            ax.text(
                0.5, 0.5,
                "Keine Produktionsdaten für diese Woche",
                transform=ax.transAxes,
                ha="center", va="center",
                color=self.styles.text_muted,
                fontsize=14,
            )
            ax.set_title("☀️ Solar & Preis Analyse", fontsize=14, color=self.styles.text_primary)
            return

        # Preise als Dict
        price_dict: dict[tuple[date, int], float] = {}
        for p in price_stats["hourly_prices"]:
            price_dict[(p.date, p.hour)] = p.price_net

        hours = []
        productions = []
        prices = []
        sizes = []

        for pred in week_predictions:
            key = (pred.target_date, pred.target_hour)
            if key in price_dict:
                hours.append(pred.target_hour)
                productions.append(pred.actual_kwh)
                prices.append(price_dict[key])
                sizes.append(max(30, pred.actual_kwh * 250))

        if not hours:
            ax.text(0.5, 0.5, "Keine überlappenden Daten", transform=ax.transAxes,
                    ha="center", va="center", color=self.styles.text_muted, fontsize=14)
            return

        # Moderne Colormap für Scatter
        scatter_cmap = LinearSegmentedColormap.from_list(
            "solar_scatter",
            [self.styles.solar_gold, self.styles.solar_orange, self.styles.neon_pink],
            N=256
        )

        # Hintergrund: Produktionszonen
        production_hours = sorted(set(hours))
        for h in production_hours:
            ax.axvspan(h - 0.4, h + 0.4, alpha=0.08, color=self.styles.solar_yellow, zorder=0)

        # Scatter mit Glow-Effekt
        # Äußerer Glow
        ax.scatter(
            hours, prices, s=[s * 1.5 for s in sizes],
            c=productions, cmap=scatter_cmap,
            alpha=0.2, edgecolors="none",
        )
        # Hauptpunkte
        scatter = ax.scatter(
            hours, prices, s=sizes,
            c=productions, cmap=scatter_cmap,
            alpha=0.85,
            edgecolors=self.styles.text_primary,
            linewidths=1,
            zorder=5,
        )

        # Durchschnittspreis mit Glow
        avg_price = np.mean(prices)
        ax.axhline(y=avg_price, color=self.styles.neon_cyan, linestyle="--",
                   linewidth=3, alpha=0.3, zorder=1)
        ax.axhline(y=avg_price, color=self.styles.neon_cyan, linestyle="--",
                   linewidth=1.5, alpha=0.9, zorder=2,
                   label=f"Ø Preis: {avg_price:.1f} ct/kWh")

        ax.set_xlabel("Stunde", fontsize=12)
        ax.set_ylabel("Strompreis (ct/kWh)", fontsize=12)
        ax.set_title(
            "☀️ Solar-Produktion & Strompreis  (Punktgröße = Produktion)",
            fontsize=14,
            fontweight="bold",
            color=self.styles.text_primary,
            pad=12,
        )

        ax.set_xlim(5.5, 19.5)
        ax.set_xticks(range(6, 20, 2))
        ax.set_xticklabels([f"{h}:00" for h in range(6, 20, 2)], fontsize=11)

        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02, aspect=25)
        cbar.set_label("Produktion (kWh)", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

        # KPI Box mit modernem Styling
        total_production = sum(productions)
        weighted_avg_price = sum(p * prod for p, prod in zip(prices, productions)) / total_production if total_production > 0 else 0
        estimated_value = total_production * weighted_avg_price / 100

        kpi_text = (
            f"Σ Produktion:  {total_production:.2f} kWh\n"
            f"Ø Einspeisepreis:  {weighted_avg_price:.1f} ct/kWh\n"
            f"💶 Geschätzter Wert:  {estimated_value:.2f} €"
        )

        props = dict(
            boxstyle="round,pad=0.6,rounding_size=0.15",
            facecolor=self.styles.background_card,
            edgecolor=self.styles.neon_cyan,
            alpha=0.95,
            linewidth=2,
        )

        ax.text(
            0.02, 0.97,
            kpi_text,
            transform=ax.transAxes,
            fontsize=11,
            color=self.styles.text_primary,
            ha="left",
            va="top",
            bbox=props,
            family="monospace",
            linespacing=1.4,
        )

        # Grid modernisieren
        ax.yaxis.grid(True, alpha=0.15, linestyle="-", linewidth=0.5)
        ax.xaxis.grid(False)

    def _add_modern_footer(self, fig: "Figure", year: int, week: int) -> None:
        """Fügt einen modernen Footer hinzu."""
        timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")

        # Linke Seite: Branding
        fig.text(
            0.06, 0.012,
            "SFML Stats",
            fontsize=11,
            color=self.styles.neon_cyan,
            ha="left",
            va="bottom",
            fontweight="bold",
            style="italic",
        )

        # Mitte: Version
        fig.text(
            0.5, 0.012,
            "v2.0 · Modern Edition",
            fontsize=9,
            color=self.styles.text_muted,
            ha="center",
            va="bottom",
        )

        # Rechte Seite: Zeitstempel
        fig.text(
            0.94, 0.012,
            f"Generiert: {timestamp}",
            fontsize=9,
            color=self.styles.text_muted,
            ha="right",
            va="bottom",
        )

        # Dezente Linie über dem Footer
        from matplotlib.lines import Line2D
        line = Line2D(
            [0.06, 0.94], [0.025, 0.025],
            transform=fig.transFigure,
            color=self.styles.border,
            linewidth=0.5,
            alpha=0.5,
        )
        fig.add_artist(line)
