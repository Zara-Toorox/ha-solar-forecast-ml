"""Chart styles for SFML Stats."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..const import COLORS

if TYPE_CHECKING:
    from matplotlib.colors import LinearSegmentedColormap


@dataclass
class ChartStyles:
    """Zentrale Style-Konfiguration für alle Charts."""

    # Hintergrundfarben
    background: str = COLORS["background"]
    background_light: str = COLORS["background_light"]
    background_card: str = COLORS["background_card"]

    # Textfarben
    text_primary: str = COLORS["text_primary"]
    text_secondary: str = COLORS["text_secondary"]
    text_muted: str = COLORS["text_muted"]

    # Akzentfarben
    solar_yellow: str = COLORS["solar_yellow"]
    solar_orange: str = COLORS["solar_orange"]
    price_green: str = COLORS["price_green"]
    price_red: str = COLORS["price_red"]
    ml_purple: str = COLORS["ml_purple"]
    rule_based_blue: str = COLORS["rule_based_blue"]

    # Chart-spezifisch
    actual: str = COLORS["actual"]
    predicted: str = COLORS["predicted"]
    accuracy_good: str = COLORS["accuracy_good"]
    accuracy_medium: str = COLORS["accuracy_medium"]
    accuracy_bad: str = COLORS["accuracy_bad"]

    # Grid und Borders
    grid: str = COLORS["grid"]
    border: str = COLORS["border"]

    # Schriftgrößen
    title_size: int = 16
    subtitle_size: int = 12
    label_size: int = 10
    tick_size: int = 9
    legend_size: int = 9

    # Schriftart
    font_family: str = "DejaVu Sans"

    def get_accuracy_color(self, accuracy: float) -> str:
        """Gibt die Farbe basierend auf der Genauigkeit zurück.

        Args:
            accuracy: Genauigkeit in Prozent (0-100+)

        Returns:
            Hex-Farbcode
        """
        if accuracy >= 80:
            return self.accuracy_good
        elif accuracy >= 50:
            return self.accuracy_medium
        else:
            return self.accuracy_bad

    def get_price_color(self, price: float, avg_price: float) -> str:
        """Gibt die Farbe basierend auf dem Preis zurück.

        Args:
            price: Aktueller Preis
            avg_price: Durchschnittspreis als Referenz

        Returns:
            Hex-Farbcode
        """
        if price < avg_price * 0.8:
            return self.price_green
        elif price > avg_price * 1.2:
            return self.price_red
        else:
            return self.solar_orange


def apply_dark_theme() -> None:
    """Wendet das Dark Theme global auf Matplotlib an.

    WICHTIG: Diese Funktion muss in einem Executor aufgerufen werden,
    da matplotlib blockierende I/O-Operationen ausführt.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    styles = ChartStyles()

    # Globale Matplotlib-Einstellungen
    plt.style.use("dark_background")

    # Detaillierte Anpassungen
    mpl.rcParams.update({
        # Hintergrund
        "figure.facecolor": styles.background,
        "axes.facecolor": styles.background_light,
        "savefig.facecolor": styles.background,

        # Text
        "text.color": styles.text_primary,
        "axes.labelcolor": styles.text_primary,
        "xtick.color": styles.text_secondary,
        "ytick.color": styles.text_secondary,

        # Grid
        "axes.edgecolor": styles.border,
        "grid.color": styles.grid,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,

        # Schriftgrößen
        "font.size": styles.label_size,
        "axes.titlesize": styles.title_size,
        "axes.labelsize": styles.label_size,
        "xtick.labelsize": styles.tick_size,
        "ytick.labelsize": styles.tick_size,
        "legend.fontsize": styles.legend_size,

        # Schriftart
        "font.family": styles.font_family,

        # Legende
        "legend.facecolor": styles.background_card,
        "legend.edgecolor": styles.border,
        "legend.framealpha": 0.9,

        # Figur
        "figure.dpi": 100,
        "savefig.dpi": 150,

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def create_price_colormap() -> "LinearSegmentedColormap":
    """Erstellt eine Colormap für Preise (grün -> gelb -> rot).

    WICHTIG: Diese Funktion muss in einem Executor aufgerufen werden.

    Returns:
        LinearSegmentedColormap für Preisvisualisierung
    """
    from matplotlib.colors import LinearSegmentedColormap

    styles = ChartStyles()
    colors = [styles.price_green, styles.solar_yellow, styles.price_red]
    return LinearSegmentedColormap.from_list("price_cmap", colors, N=256)


def create_accuracy_colormap() -> "LinearSegmentedColormap":
    """Erstellt eine Colormap für Genauigkeit (rot -> gelb -> grün).

    WICHTIG: Diese Funktion muss in einem Executor aufgerufen werden.

    Returns:
        LinearSegmentedColormap für Genauigkeitsvisualisierung
    """
    from matplotlib.colors import LinearSegmentedColormap

    styles = ChartStyles()
    colors = [styles.accuracy_bad, styles.accuracy_medium, styles.accuracy_good]
    return LinearSegmentedColormap.from_list("accuracy_cmap", colors, N=256)


def create_solar_colormap() -> "LinearSegmentedColormap":
    """Erstellt eine Colormap für Solarproduktion (dunkel -> gelb -> orange).

    WICHTIG: Diese Funktion muss in einem Executor aufgerufen werden.

    Returns:
        LinearSegmentedColormap für Solarvisualisierung
    """
    from matplotlib.colors import LinearSegmentedColormap

    styles = ChartStyles()
    colors = [styles.background_light, styles.solar_yellow, styles.solar_orange]
    return LinearSegmentedColormap.from_list("solar_cmap", colors, N=256)


# Vordefinierte Farbpaletten für verschiedene Anwendungsfälle
COLOR_PALETTE_SOLAR = [
    COLORS["solar_yellow"],
    COLORS["solar_orange"],
    "#ff5722",  # Deep Orange
    "#e91e63",  # Pink
]

COLOR_PALETTE_COMPARISON = [
    COLORS["predicted"],  # Blau für Vorhersage
    COLORS["actual"],  # Grün für Actual
    COLORS["ml_purple"],  # Lila für ML
    COLORS["rule_based_blue"],  # Hellblau für Rule-Based
]

COLOR_PALETTE_PRICES = [
    COLORS["price_green"],
    COLORS["solar_yellow"],
    COLORS["solar_orange"],
    COLORS["price_red"],
]

# Wochentage auf Deutsch
WEEKDAY_NAMES_DE = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
WEEKDAY_NAMES_FULL_DE = [
    "Montag", "Dienstag", "Mittwoch", "Donnerstag",
    "Freitag", "Samstag", "Sonntag"
]

# Monate auf Deutsch
MONTH_NAMES_DE = [
    "Januar", "Februar", "März", "April", "Mai", "Juni",
    "Juli", "August", "September", "Oktober", "November", "Dezember"
]
MONTH_NAMES_SHORT_DE = [
    "Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
    "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"
]
