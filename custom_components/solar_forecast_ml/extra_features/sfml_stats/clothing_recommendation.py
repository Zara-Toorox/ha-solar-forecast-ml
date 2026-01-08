# ******************************************************************************
# @copyright (C) 2025 Zara-Toorox - SFML Stats
# * This program is protected by a Proprietary Non-Commercial License.
# 1. Personal and Educational use only.
# 2. COMMERCIAL USE AND AI TRAINING ARE STRICTLY PROHIBITED.
# 3. Clear attribution to "Zara-Toorox" is required.
# * Full license terms: https://github.com/Zara-Toorox/sfml-stats/blob/main/LICENSE
# ******************************************************************************

"""Clothing recommendation engine based on weather data."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

_LOGGER = logging.getLogger(__name__)


@dataclass
class ClothingRecommendation:
    """Data class for clothing recommendation."""

    unterbekleidung: str
    unterbekleidung_icon: str
    oberbekleidung: str
    oberbekleidung_icon: str
    jacke: str
    jacke_icon: str
    kopfbedeckung: str
    kopfbedeckung_icon: str
    zusaetze: list[str]
    zusaetze_icons: list[str]
    text_de: str
    text_en: str
    weather_summary: dict[str, Any]


# Clothing options with icons
CLOTHING_OPTIONS = {
    "unterbekleidung": {
        "kurze_hose": ("Kurze Hose", "Shorts", "🩳"),
        "lange_hose": ("Lange Hose", "Long pants", "👖"),
        "jeans": ("Jeans", "Jeans", "👖"),
        "stoffhose": ("Stoffhose", "Dress pants", "👖"),
        "jogginghose": ("Jogginghose", "Sweatpants", "🏃"),
    },
    "oberbekleidung": {
        "tshirt": ("T-Shirt", "T-Shirt", "👕"),
        "hemd": ("Hemd", "Shirt", "👔"),
        "pullover": ("Pullover", "Sweater", "🥼"),
        "longsleeve": ("Langarmshirt", "Long sleeve", "👕"),
        "hoodie": ("Hoodie", "Hoodie", "🎽"),
    },
    "jacke": {
        "keine": ("Keine Jacke", "No jacket", "🚫"),
        "leichte_windjacke": ("Leichte Windjacke", "Light windbreaker", "🌬️"),
        "uebergangsjacke": ("Übergangsjacke", "Light jacket", "🧥"),
        "regenjacke": ("Regenjacke", "Rain jacket", "🌧️"),
        "winterjacke": ("Winterjacke", "Winter jacket", "🥶"),
        "softshelljacke": ("Softshelljacke", "Softshell jacket", "🧥"),
    },
    "kopfbedeckung": {
        "keine": ("Keine", "None", "🚫"),
        "cap": ("Cap", "Cap", "🧢"),
        "muetze": ("Mütze", "Beanie", "🧶"),
        "sonnenhut": ("Sonnenhut", "Sun hat", "👒"),
    },
    "zusaetze": {
        "keine": ("Keine", "None", "🚫"),
        "regenschirm": ("Regenschirm", "Umbrella", "☂️"),
        "sonnenbrille": ("Sonnenbrille", "Sunglasses", "🕶️"),
        "sonnencreme": ("Sonnencreme", "Sunscreen", "🧴"),
        "handschuhe": ("Handschuhe", "Gloves", "🧤"),
        "schal": ("Schal", "Scarf", "🧣"),
    },
}


def get_recommendation(weather_data: dict[str, Any], forecast_hours: list[dict] | None = None) -> ClothingRecommendation:
    """Generate clothing recommendation based on weather data.

    Args:
        weather_data: Current weather conditions with keys:
            - temperature: Current temp in °C
            - humidity: Humidity in %
            - wind_speed: Wind speed in m/s or km/h
            - precipitation: Precipitation in mm
            - cloud_cover: Cloud cover in %
            - pressure: Pressure in hPa
            - uv_index: UV index (optional)
            - radiation: Solar radiation W/m² (optional)
        forecast_hours: List of hourly forecasts for the day (optional)

    Returns:
        ClothingRecommendation with all clothing items and text descriptions
    """
    temp = weather_data.get("temperature", 15)
    humidity = weather_data.get("humidity", 50)
    wind_speed = weather_data.get("wind_speed", 0)
    precipitation = weather_data.get("precipitation", 0)
    cloud_cover = weather_data.get("cloud_cover", 50)
    uv_index = weather_data.get("uv_index", 0)
    radiation = weather_data.get("radiation", 0)

    # Calculate rain probability from forecast if available
    rain_prob = 0
    if forecast_hours:
        # Get max rain probability for next 8 hours
        for hour in forecast_hours[:8]:
            hour_rain_prob = hour.get("precipitation_probability", 0)
            if hour_rain_prob and hour_rain_prob > rain_prob:
                rain_prob = hour_rain_prob

    # If no UV index but we have radiation, estimate UV
    if uv_index == 0 and radiation > 0:
        # Rough estimate: UV index ≈ radiation / 100 (simplified)
        uv_index = min(11, radiation / 100)

    # Calculate "feels like" temperature (wind chill / heat index simplified)
    feels_like = temp
    if temp < 10 and wind_speed > 5:
        # Wind chill effect
        feels_like = temp - (wind_speed * 0.3)
    elif temp > 25 and humidity > 60:
        # Heat index effect
        feels_like = temp + ((humidity - 60) * 0.1)

    _LOGGER.debug(
        "Clothing recommendation input: temp=%.1f, feels_like=%.1f, humidity=%d, "
        "wind=%.1f, rain_prob=%d, uv=%d, clouds=%d",
        temp, feels_like, humidity, wind_speed, rain_prob, uv_index, cloud_cover
    )

    # Decision logic for each category
    unterbekleidung = _get_unterbekleidung(feels_like, temp)
    oberbekleidung = _get_oberbekleidung(feels_like, temp)
    jacke = _get_jacke(feels_like, temp, rain_prob, precipitation, wind_speed)
    kopfbedeckung = _get_kopfbedeckung(feels_like, temp, uv_index, radiation, cloud_cover)
    zusaetze = _get_zusaetze(temp, rain_prob, precipitation, uv_index, radiation, feels_like)

    # Get display names and icons
    unter_de, unter_en, unter_icon = CLOTHING_OPTIONS["unterbekleidung"][unterbekleidung]
    ober_de, ober_en, ober_icon = CLOTHING_OPTIONS["oberbekleidung"][oberbekleidung]
    jacke_de, jacke_en, jacke_icon = CLOTHING_OPTIONS["jacke"][jacke]
    kopf_de, kopf_en, kopf_icon = CLOTHING_OPTIONS["kopfbedeckung"][kopfbedeckung]

    zusaetze_de = []
    zusaetze_en = []
    zusaetze_icons = []
    for z in zusaetze:
        z_de, z_en, z_icon = CLOTHING_OPTIONS["zusaetze"][z]
        zusaetze_de.append(z_de)
        zusaetze_en.append(z_en)
        zusaetze_icons.append(z_icon)

    # Generate natural language text
    text_de = _generate_text_de(
        temp, feels_like, wind_speed, rain_prob, uv_index,
        unter_de, ober_de, jacke_de, kopf_de, zusaetze_de
    )
    text_en = _generate_text_en(
        temp, feels_like, wind_speed, rain_prob, uv_index,
        unter_en, ober_en, jacke_en, kopf_en, zusaetze_en
    )

    return ClothingRecommendation(
        unterbekleidung=unter_de,
        unterbekleidung_icon=unter_icon,
        oberbekleidung=ober_de,
        oberbekleidung_icon=ober_icon,
        jacke=jacke_de,
        jacke_icon=jacke_icon,
        kopfbedeckung=kopf_de,
        kopfbedeckung_icon=kopf_icon,
        zusaetze=zusaetze_de,
        zusaetze_icons=zusaetze_icons,
        text_de=text_de,
        text_en=text_en,
        weather_summary={
            "temperature": round(temp, 1),
            "feels_like": round(feels_like, 1),
            "humidity": round(humidity),
            "wind_speed": round(wind_speed, 1),
            "rain_probability": round(rain_prob),
            "uv_index": round(uv_index),
            "cloud_cover": round(cloud_cover),
        }
    )


def _get_unterbekleidung(feels_like: float, temp: float) -> str:
    """Determine lower body clothing."""
    if feels_like >= 25 or temp >= 26:
        return "kurze_hose"
    elif feels_like >= 20:
        return "stoffhose"  # Lighter pants for mild weather
    elif feels_like >= 10:
        return "jeans"
    else:
        return "lange_hose"


def _get_oberbekleidung(feels_like: float, temp: float) -> str:
    """Determine upper body clothing."""
    if feels_like >= 26 or temp >= 27:
        return "tshirt"
    elif feels_like >= 20:
        return "longsleeve"
    elif feels_like >= 12:
        return "hoodie"
    elif feels_like >= 5:
        return "pullover"
    else:
        return "pullover"  # Heavy sweater for cold


def _get_jacke(feels_like: float, temp: float, rain_prob: float, precipitation: float, wind_speed: float) -> str:
    """Determine jacket type."""
    # Winter jacket for very cold
    if feels_like < 0 or temp < 2:
        return "winterjacke"

    # Rain jacket if rain expected
    if rain_prob > 60 or precipitation > 0.5:
        if feels_like < 10:
            return "regenjacke"  # Cold + rain
        return "regenjacke"

    # Transition jacket for cool weather
    if feels_like < 10:
        return "uebergangsjacke"

    # Light windbreaker for windy conditions
    if wind_speed > 8 and feels_like < 18:
        return "leichte_windjacke"

    # Softshell for mild but cool
    if feels_like < 15:
        return "softshelljacke"

    # No jacket needed for warm weather
    if feels_like >= 22 and rain_prob < 30:
        return "keine"

    # Light jacket for in-between
    if feels_like < 20:
        return "leichte_windjacke"

    return "keine"


def _get_kopfbedeckung(feels_like: float, temp: float, uv_index: float, radiation: float, cloud_cover: float) -> str:
    """Determine headwear."""
    # Beanie for cold
    if feels_like < 5 or temp < 3:
        return "muetze"

    # Cap/Sun hat for high UV or sunny conditions
    if uv_index >= 6 or (radiation > 600 and cloud_cover < 30):
        return "cap"

    # Sun hat for very high UV
    if uv_index >= 8:
        return "sonnenhut"

    # Cap for moderate sun exposure
    if radiation > 400 and cloud_cover < 50:
        return "cap"

    return "keine"


def _get_zusaetze(temp: float, rain_prob: float, precipitation: float, uv_index: float, radiation: float, feels_like: float) -> list[str]:
    """Determine accessories."""
    zusaetze = []

    # Umbrella for rain
    if rain_prob > 50 or precipitation > 0.3:
        zusaetze.append("regenschirm")

    # Sunglasses for sunny days
    if uv_index >= 4 or radiation > 400:
        zusaetze.append("sonnenbrille")

    # Sunscreen for high UV
    if uv_index >= 6:
        zusaetze.append("sonnencreme")

    # Gloves for cold
    if feels_like < 2 or temp < 0:
        zusaetze.append("handschuhe")

    # Scarf for cold
    if feels_like < 0 or temp < -2:
        zusaetze.append("schal")

    if not zusaetze:
        zusaetze.append("keine")

    return zusaetze


def _generate_text_de(
    temp: float, feels_like: float, wind_speed: float, rain_prob: float, uv_index: float,
    unter: str, ober: str, jacke: str, kopf: str, zusaetze: list[str]
) -> str:
    """Generate German recommendation text."""
    # Weather description
    if temp < 0:
        weather_desc = f"Heute wird es frostig kalt mit {temp:.0f}°C"
    elif temp < 10:
        weather_desc = f"Heute wird es kühl mit {temp:.0f}°C"
    elif temp < 18:
        weather_desc = f"Heute wird es mild mit {temp:.0f}°C"
    elif temp < 25:
        weather_desc = f"Heute wird es angenehm warm mit {temp:.0f}°C"
    else:
        weather_desc = f"Heute wird es heiß mit {temp:.0f}°C"

    # Add wind info if significant
    if wind_speed > 10:
        weather_desc += f" und starkem Wind ({wind_speed:.0f} km/h)"
    elif wind_speed > 5:
        weather_desc += f" und leichtem Wind"

    # Feels like difference
    if abs(feels_like - temp) > 3:
        weather_desc += f" (gefühlt {feels_like:.0f}°C)"

    weather_desc += "."

    # Clothing recommendation
    clothing_parts = []

    # Lower body
    clothing_parts.append(f"Zieh dir eine {unter} an")

    # Upper body
    if ober == "T-Shirt":
        clothing_parts.append(f"ein {ober}")
    else:
        clothing_parts.append(f"einen {ober}")

    # Jacket
    if jacke != "Keine Jacke":
        clothing_parts.append(f"und nimm eine {jacke} mit")

    clothing_text = ", ".join(clothing_parts[:2])
    if len(clothing_parts) > 2:
        clothing_text += " " + clothing_parts[2]
    clothing_text += "."

    # Headwear
    head_text = ""
    if kopf == "Mütze":
        head_text = " Eine Mütze hält deine Ohren warm."
    elif kopf == "Cap" or kopf == "Sonnenhut":
        head_text = f" Ein {kopf} schützt dich vor der Sonne."

    # Accessories
    zusaetze_text = ""
    filtered_zusaetze = [z for z in zusaetze if z != "Keine"]
    if filtered_zusaetze:
        if len(filtered_zusaetze) == 1:
            zusaetze_text = f" Denk auch an: {filtered_zusaetze[0]}."
        else:
            zusaetze_text = f" Denk auch an: {', '.join(filtered_zusaetze[:-1])} und {filtered_zusaetze[-1]}."

    # Rain warning
    rain_text = ""
    if rain_prob > 70:
        rain_text = " Es wird sehr wahrscheinlich regnen!"
    elif rain_prob > 50:
        rain_text = " Regen ist möglich."

    return weather_desc + " " + clothing_text + head_text + zusaetze_text + rain_text


def _generate_text_en(
    temp: float, feels_like: float, wind_speed: float, rain_prob: float, uv_index: float,
    unter: str, ober: str, jacke: str, kopf: str, zusaetze: list[str]
) -> str:
    """Generate English recommendation text."""
    # Weather description
    if temp < 0:
        weather_desc = f"Today will be freezing cold at {temp:.0f}°C"
    elif temp < 10:
        weather_desc = f"Today will be chilly at {temp:.0f}°C"
    elif temp < 18:
        weather_desc = f"Today will be mild at {temp:.0f}°C"
    elif temp < 25:
        weather_desc = f"Today will be pleasantly warm at {temp:.0f}°C"
    else:
        weather_desc = f"Today will be hot at {temp:.0f}°C"

    # Add wind info if significant
    if wind_speed > 10:
        weather_desc += f" with strong winds ({wind_speed:.0f} km/h)"
    elif wind_speed > 5:
        weather_desc += " with light winds"

    # Feels like difference
    if abs(feels_like - temp) > 3:
        weather_desc += f" (feels like {feels_like:.0f}°C)"

    weather_desc += "."

    # Clothing recommendation
    clothing_text = f"Wear {unter.lower()}, a {ober.lower()}"

    # Jacket
    if jacke != "No jacket":
        clothing_text += f", and bring a {jacke.lower()}"

    clothing_text += "."

    # Headwear
    head_text = ""
    if kopf == "Beanie":
        head_text = " A beanie will keep your ears warm."
    elif kopf in ["Cap", "Sun hat"]:
        head_text = f" A {kopf.lower()} will protect you from the sun."

    # Accessories
    zusaetze_text = ""
    filtered_zusaetze = [z for z in zusaetze if z != "None"]
    if filtered_zusaetze:
        if len(filtered_zusaetze) == 1:
            zusaetze_text = f" Don't forget: {filtered_zusaetze[0].lower()}."
        else:
            zusaetze_text = f" Don't forget: {', '.join(z.lower() for z in filtered_zusaetze[:-1])} and {filtered_zusaetze[-1].lower()}."

    # Rain warning
    rain_text = ""
    if rain_prob > 70:
        rain_text = " Rain is very likely!"
    elif rain_prob > 50:
        rain_text = " Rain is possible."

    return weather_desc + " " + clothing_text + head_text + zusaetze_text + rain_text
