# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - Solar Forecast ML
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/ha-solar-forecast-ml/blob/main/LICENSE
# ******************************************************************************

"""Chart styles for SFML Stats - Modern Redesign."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..const import COLORS

if TYPE_CHECKING:
    from matplotlib.colors import LinearSegmentedColormap


@dataclass
class ChartStyles:
    """Zentrale Style-Konfiguration für alle Charts - Modern Edition."""

    # Hintergrundfarben (tieferes, moderneres Dark Theme)
    background: str = COLORS["background"]
    background_light: str = COLORS["background_light"]
    background_card: str = COLORS["background_card"]
    background_card_hover: str = COLORS["background_card_hover"]

    # Textfarben (besserer Kontrast)
    text_primary: str = COLORS["text_primary"]
    text_secondary: str = COLORS["text_secondary"]
    text_muted: str = COLORS["text_muted"]

    # Solar-Farben (lebendiger)
    solar_yellow: str = COLORS["solar_yellow"]
    solar_orange: str = COLORS["solar_orange"]
    solar_gold: str = COLORS["solar_gold"]

    # Neon-Akzente für Highlights
    neon_cyan: str = COLORS["neon_cyan"]
    neon_green: str = COLORS["neon_green"]
    neon_pink: str = COLORS["neon_pink"]
    neon_purple: str = COLORS["neon_purple"]

    # Preisfarben
    price_green: str = COLORS["price_green"]
    price_red: str = COLORS["price_red"]
    price_yellow: str = COLORS["price_yellow"]

    # ML Farben
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
    border_glow: str = COLORS["border_glow"]

    # Gradient-Endpunkte
    gradient_start: str = COLORS["gradient_start"]
    gradient_end: str = COLORS["gradient_end"]

    # Schriftgrößen (etwas größer für bessere Lesbarkeit)
    title_size: int = 18
    subtitle_size: int = 13
    label_size: int = 11
    tick_size: int = 10
    legend_size: int = 10
    kpi_value_size: int = 20
    kpi_label_size: int = 11

    # Schriftart
    font_family: str = "DejaVu Sans"

    # Moderne Effekt-Parameter
    glow_alpha: float = 0.3
    glow_linewidth: float = 8.0
    shadow_alpha: float = 0.4
    glass_alpha: float = 0.15
    bar_radius: float = 0.3  # Für abgerundete Balken

    # Animation-ähnliche Effekte (für statische Bilder simuliert)
    gradient_steps: int = 50

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
    """Wendet das Dark Theme global auf Matplotlib an - Modern Edition.

    WICHTIG: Diese Funktion muss in einem Executor aufgerufen werden,
    da matplotlib blockierende I/O-Operationen ausführt.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    styles = ChartStyles()

    # Globale Matplotlib-Einstellungen
    plt.style.use("dark_background")

    # Detaillierte Anpassungen für modernes Design
    mpl.rcParams.update({
        # Hintergrund (tieferes Schwarz)
        "figure.facecolor": styles.background,
        "axes.facecolor": styles.background_light,
        "savefig.facecolor": styles.background,

        # Text (besserer Kontrast)
        "text.color": styles.text_primary,
        "axes.labelcolor": styles.text_primary,
        "xtick.color": styles.text_secondary,
        "ytick.color": styles.text_secondary,

        # Grid (subtiler, moderner)
        "axes.edgecolor": styles.border,
        "grid.color": styles.grid,
        "grid.alpha": 0.2,
        "axes.grid": True,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,

        # Schriftgrößen (größer für bessere Lesbarkeit)
        "font.size": styles.label_size,
        "axes.titlesize": styles.title_size,
        "axes.labelsize": styles.label_size,
        "xtick.labelsize": styles.tick_size,
        "ytick.labelsize": styles.tick_size,
        "legend.fontsize": styles.legend_size,

        # Schriftart
        "font.family": styles.font_family,
        "font.weight": "normal",

        # Legende (glassmorphism-ähnlich)
        "legend.facecolor": styles.background_card,
        "legend.edgecolor": styles.border,
        "legend.framealpha": 0.85,
        "legend.borderpad": 0.8,
        "legend.labelspacing": 0.6,

        # Figur
        "figure.dpi": 100,
        "savefig.dpi": 150,

        # Spines (alle aus für cleaneres Design)
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,

        # Lines
        "lines.linewidth": 2.0,
        "lines.antialiased": True,

        # Patches (für Balken etc.)
        "patch.linewidth": 0.5,
        "patch.antialiased": True,
    })


def create_gradient_image(
    width: int,
    height: int,
    color_start: str,
    color_end: str,
    direction: str = "horizontal"
) -> "np.ndarray":
    """Erstellt ein Gradient-Bild für Hintergründe.

    Args:
        width: Breite in Pixeln
        height: Höhe in Pixeln
        color_start: Start-Farbe (hex)
        color_end: End-Farbe (hex)
        direction: "horizontal" oder "vertical"

    Returns:
        NumPy Array mit RGBA-Werten
    """
    import numpy as np
    from matplotlib.colors import to_rgba

    start_rgba = to_rgba(color_start)
    end_rgba = to_rgba(color_end)

    if direction == "horizontal":
        gradient = np.linspace(0, 1, width)
        gradient = np.tile(gradient, (height, 1))
    else:
        gradient = np.linspace(0, 1, height)
        gradient = np.tile(gradient.reshape(-1, 1), (1, width))

    # Interpolieren zwischen den Farben
    result = np.zeros((height, width, 4))
    for i in range(4):
        result[:, :, i] = start_rgba[i] + gradient * (end_rgba[i] - start_rgba[i])

    return result


def add_glow_effect(ax, x, y, color: str, alpha: float = 0.3, linewidth: float = 8.0) -> None:
    """Fügt einen Glow-Effekt zu einer Linie hinzu.

    Args:
        ax: Matplotlib Axes
        x: X-Werte
        y: Y-Werte
        color: Linienfarbe
        alpha: Transparenz des Glows
        linewidth: Breite des Glow-Effekts
    """
    # Mehrere Linien mit abnehmender Transparenz für Glow
    for lw, a in [(linewidth, alpha * 0.2), (linewidth * 0.6, alpha * 0.4), (linewidth * 0.3, alpha * 0.6)]:
        ax.plot(x, y, color=color, linewidth=lw, alpha=a, solid_capstyle="round")


def draw_rounded_bar(ax, x, height, width, color, radius: float = 0.3, **kwargs) -> None:
    """Zeichnet einen Balken mit abgerundeten Ecken.

    Args:
        ax: Matplotlib Axes
        x: X-Position (Mitte des Balkens)
        height: Höhe des Balkens
        width: Breite des Balkens
        color: Farbe
        radius: Rundungsradius (0-0.5)
        **kwargs: Weitere Parameter für FancyBboxPatch
    """
    from matplotlib.patches import FancyBboxPatch

    # Position berechnen (x ist Mitte, wir brauchen linke untere Ecke)
    left = x - width / 2
    bottom = 0

    # Rundung begrenzen
    radius = min(radius, 0.5)
    pad = radius * min(width, height) * 0.5

    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={pad}",
        facecolor=color,
        edgecolor="none",
        **kwargs
    )
    ax.add_patch(patch)


def draw_glass_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    text_color: str,
    border_color: str = None,
    fontsize: int = 12,
) -> None:
    """Zeichnet eine Box mit Glassmorphism-Effekt.

    Args:
        ax: Matplotlib Axes
        x, y: Position (Transform-Koordinaten)
        width, height: Größe
        text: Anzuzeigender Text
        text_color: Textfarbe
        border_color: Rahmenfarbe (optional)
        fontsize: Schriftgröße
    """
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Bbox

    styles = ChartStyles()

    # Semi-transparenter Hintergrund
    props = dict(
        boxstyle="round,pad=0.5,rounding_size=0.2",
        facecolor=styles.background_card,
        edgecolor=border_color or styles.border,
        alpha=0.85,
        linewidth=1.5,
    )

    ax.text(
        x, y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=text_color,
        ha="center",
        va="center",
        bbox=props,
        fontweight="bold",
    )


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
