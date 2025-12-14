"""Weekly report chart for SFML Stats."""
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
        """Synchrones Chart-Rendering - läuft im Executor."""
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from .styles import apply_dark_theme

        apply_dark_theme()

        # Figure mit GridSpec erstellen
        fig = plt.figure(figsize=self._figsize, facecolor=self.styles.background)

        # GridSpec Layout: 4 Zeilen, 2 Spalten
        gs = gridspec.GridSpec(
            4, 2,
            figure=fig,
            height_ratios=[0.8, 1, 1, 1.2],
            width_ratios=[1, 1],
            hspace=0.35,
            wspace=0.25,
        )

        # Header (ganze Breite)
        ax_header = fig.add_subplot(gs[0, :])
        self._draw_header(ax_header, year, week, week_start, week_end, solar_stats, price_stats)

        # Chart 1: Produktion vs. Vorhersage (links oben)
        ax_production = fig.add_subplot(gs[1, 0])
        self._draw_production_chart(ax_production, solar_stats)

        # Chart 2: ML vs. Rule-Based (rechts oben)
        ax_ml = fig.add_subplot(gs[1, 1])
        self._draw_ml_contribution_chart(ax_ml, solar_stats)

        # Chart 3: Preis-Heatmap (links mitte)
        ax_price = fig.add_subplot(gs[2, 0])
        self._draw_price_heatmap(ax_price, price_stats)

        # Chart 4: Genauigkeit Heatmap (rechts mitte)
        ax_accuracy = fig.add_subplot(gs[2, 1])
        self._draw_accuracy_heatmap(ax_accuracy, solar_stats)

        # Chart 5: Solar + Preis Korrelation (ganze Breite unten)
        ax_correlation = fig.add_subplot(gs[3, :])
        self._draw_solar_price_correlation_sync(
            ax_correlation, solar_stats, price_stats,
            year, week, hourly_predictions
        )

        # Footer
        self._add_footer(fig, f"Wochenbericht KW {week}/{year}")

        return fig

    def _get_week_start(self, year: int, week: int) -> date:
        """Berechnet den Montag der angegebenen Kalenderwoche."""
        jan4 = date(year, 1, 4)
        start_of_week1 = jan4 - timedelta(days=jan4.weekday())
        return start_of_week1 + timedelta(weeks=week - 1)

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
