#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

LANG = "en"

TEXTS = {
    "welcome": {
        "de": """
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║     ██████╗ ██████╗ ██╗      █████╗ ██████╗                               ║
  ║    ██╔════╝██╔═══██╗██║     ██╔══██╗██╔══██╗                              ║
  ║    ╚█████╗ ██║   ██║██║     ███████║██████╔╝                              ║
  ║     ╚═══██╗██║   ██║██║     ██╔══██║██╔══██╗                              ║
  ║    ██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║                              ║
  ║    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                              ║
  ║                                                                           ║
  ║    ███████╗ ██████╗ ██████╗ ███████╗ ██████╗ █████╗ ███████╗████████╗     ║
  ║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝╚══██╔══╝     ║
  ║    █████╗  ██║   ██║██████╔╝█████╗  ██║     ███████║███████╗   ██║        ║
  ║    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██║     ██╔══██║╚════██║   ██║        ║
  ║    ██║     ╚██████╔╝██║  ██║███████╗╚██████╗██║  ██║███████║   ██║        ║
  ║    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝        ║
  ║                                                                           ║
  ║              ☀️  KI SENSOR EINRICHTUNGSHILFE  ☀️                           ║
  ║                                                                           ║
  ║    Dieses Tool hilft dir, die richtigen Sensoren für deine               ║
  ║    Solar Forecast Integration zu finden und einzurichten.                ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
""",
        "en": """
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║     ██████╗ ██████╗ ██╗      █████╗ ██████╗                               ║
  ║    ██╔════╝██╔═══██╗██║     ██╔══██╗██╔══██╗                              ║
  ║    ╚█████╗ ██║   ██║██║     ███████║██████╔╝                              ║
  ║     ╚═══██╗██║   ██║██║     ██╔══██║██╔══██╗                              ║
  ║    ██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║                              ║
  ║    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                              ║
  ║                                                                           ║
  ║    ███████╗ ██████╗ ██████╗ ███████╗ ██████╗ █████╗ ███████╗████████╗     ║
  ║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝╚══██╔══╝     ║
  ║    █████╗  ██║   ██║██████╔╝█████╗  ██║     ███████║███████╗   ██║        ║
  ║    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██║     ██╔══██║╚════██║   ██║        ║
  ║    ██║     ╚██████╔╝██║  ██║███████╗╚██████╗██║  ██║███████║   ██║        ║
  ║    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝        ║
  ║                                                                           ║
  ║               ☀️  AI SENSOR SETUP HELPER  ☀️                               ║
  ║                                                                           ║
  ║    This tool helps you find and configure the right sensors              ║
  ║    for your Solar Forecast integration.                                  ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
"""
    },
    "main_menu_title": {
        "de": "HAUPTMENÜ - Was möchtest du wissen?",
        "en": "MAIN MENU - What would you like to know?"
    },
    "mandatory": {
        "de": "PFLICHT-SENSOREN",
        "en": "REQUIRED SENSORS"
    },
    "optional": {
        "de": "OPTIONALE SENSOREN (nur lokale Hardware!)",
        "en": "OPTIONAL SENSORS (local hardware only!)"
    },
    "guides": {
        "de": "ANLEITUNGEN & KONFIGURATION",
        "en": "GUIDES & CONFIGURATION"
    },
    "mandatory_tag": {
        "de": "PFLICHT",
        "en": "REQUIRED"
    },
    "optional_tag": {
        "de": "OPTIONAL - Nur lokale Sensoren!",
        "en": "OPTIONAL - Local sensors only!"
    },
    "guide_tag": {
        "de": "ANLEITUNG",
        "en": "GUIDE"
    },
    "exit": {
        "de": "Beenden",
        "en": "Exit"
    },
    "press_enter": {
        "de": "\n  ◀ Drücke ENTER um zum Menü zurückzukehren...",
        "en": "\n  ◀ Press ENTER to return to menu..."
    },
    "invalid_choice": {
        "de": "⚠️  Ungültige Auswahl. Bitte erneut versuchen.",
        "en": "⚠️  Invalid choice. Please try again."
    },
    "goodbye": {
        "de": """
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║   ☀️  Viel Erfolg mit Solar Forecast!                                      ║
  ║                                                                           ║
  ║   Möge die Sonne scheinen und deine Panels glühen!  ☀️                     ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
""",
        "en": """
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║   ☀️  Good luck with Solar Forecast!                                       ║
  ║                                                                           ║
  ║   May the sun shine and your panels glow!  ☀️                              ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
"""
    },
    "what_is": {
        "de": "WAS IST DAS?",
        "en": "WHAT IS IT?"
    },
    "why_important": {
        "de": "WARUM IST DAS WICHTIG?",
        "en": "WHY IS THIS IMPORTANT?"
    },
    "unit": {
        "de": "ERWARTETE EINHEIT",
        "en": "EXPECTED UNIT"
    },
    "typical_entities": {
        "de": "TYPISCHE ENTITY-IDs",
        "en": "TYPICAL ENTITY IDs"
    },
    "common_errors": {
        "de": "HÄUFIGE FEHLER & LÖSUNGEN",
        "en": "COMMON MISTAKES & SOLUTIONS"
    },
    "tips": {
        "de": "PROFI-TIPPS",
        "en": "PRO TIPS"
    },
    "example": {
        "de": "YAML BEISPIEL",
        "en": "YAML EXAMPLE"
    },
    "local_sensor_warning": {
        "de": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  ⚠️  WICHTIG: Nur LOKALE Sensoren verwenden!                                ║
  ║                                                                            ║
  ║  Wetter-Apps und Online-Dienste liefern keine lokalen Daten!               ║
  ║  Nur echte Hardware-Sensoren an deinem Standort sind geeignet.             ║
  ╚════════════════════════════════════════════════════════════════════════════╝
""",
        "en": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  ⚠️  IMPORTANT: Use LOCAL sensors only!                                     ║
  ║                                                                            ║
  ║  Weather apps and online services don't provide local data!                ║
  ║  Only real hardware sensors at your location are suitable.                 ║
  ╚════════════════════════════════════════════════════════════════════════════╝
"""
    },
    "dc_warning": {
        "de": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  ⚡ WICHTIG: Nur DC-Leistung (Gleichstrom) verwenden!                       ║
  ║                                                                            ║
  ║  Die KI benötigt die DC-Leistung DIREKT von den Solarpanels!               ║
  ║  NIEMALS die AC-Ausgangsleistung des Wechselrichters nutzen!               ║
  ║                                                                            ║
  ║  DC = Eingang vom Dach  ✓          AC = Ausgang zum Netz  ✗                ║
  ╚════════════════════════════════════════════════════════════════════════════╝
""",
        "en": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  ⚡ IMPORTANT: Use DC power (direct current) only!                          ║
  ║                                                                            ║
  ║  The AI needs DC power DIRECTLY from the solar panels!                     ║
  ║  NEVER use the AC output power from the inverter!                          ║
  ║                                                                            ║
  ║  DC = Input from roof  ✓           AC = Output to grid  ✗                  ║
  ╚════════════════════════════════════════════════════════════════════════════╝
"""
    },
    "battery_info": {
        "de": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  🔋 WICHTIG FÜR ZERO-EXPORT ANLAGEN!                                       ║
  ║                                                                            ║
  ║  Wenn deine Anlage NICHT ins Netz einspeist, wird überschüssiger Strom     ║
  ║  gedrosselt. Die KI sieht dann nicht die echte Produktion!                 ║
  ║                                                                            ║
  ║  Dieser Sensor hilft der KI, das WAHRE Potential zu berechnen.             ║
  ╚════════════════════════════════════════════════════════════════════════════╝
""",
        "en": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  🔋 IMPORTANT FOR ZERO-EXPORT SYSTEMS!                                     ║
  ║                                                                            ║
  ║  If your system does NOT feed into the grid, excess power is throttled.    ║
  ║  The AI then cannot see the real production!                               ║
  ║                                                                            ║
  ║  This sensor helps the AI calculate the TRUE potential.                    ║
  ╚════════════════════════════════════════════════════════════════════════════╝
"""
    },
    "kwp_info": {
        "de": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  ☀️  KILOWATT-PEAK (kWp) = Leistung der SOLARZELLEN                         ║
  ║                                                                            ║
  ║  Die kWp ist die Spitzenleistung deiner PANELS unter Testbedingungen.      ║
  ║  Das ist NICHT die Leistung des Wechselrichters!                           ║
  ║                                                                            ║
  ║  Berechnung: Anzahl Module × Wp pro Modul ÷ 1000 = kWp                     ║
  ╚════════════════════════════════════════════════════════════════════════════╝
""",
        "en": """
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  ☀️  KILOWATT-PEAK (kWp) = Power of your SOLAR PANELS                       ║
  ║                                                                            ║
  ║  kWp is the peak power of your PANELS under test conditions.               ║
  ║  This is NOT the inverter power!                                           ║
  ║                                                                            ║
  ║  Calculation: Number of modules × Wp per module ÷ 1000 = kWp               ║
  ╚════════════════════════════════════════════════════════════════════════════╝
"""
    }
}

SENSORS = {
    "kwp": {
        "name": {"de": "Anlagenleistung (kWp)", "en": "System Power (kWp)"},
        "mandatory": True,
        "show_kwp_info": True,
        "what": {
            "de": """Die Anlagenleistung in Kilowatt-Peak (kWp) ist die INSTALLIERTE
Leistung deiner SOLARZELLEN - NICHT des Wechselrichters!

  ┌─────────────────────────────────────────────────────────────────────┐
  │  kWp = Kilowatt-Peak = maximale Leistung unter Standardbedingungen  │
  │                                                                     │
  │  Standard-Testbedingungen (STC):                                    │
  │    • 1000 W/m² Sonneneinstrahlung                                   │
  │    • 25°C Zelltemperatur                                            │
  │    • AM 1.5 Spektrum                                                │
  └─────────────────────────────────────────────────────────────────────┘

⚠️  WICHTIG: Die kWp-Angabe findest du auf dem Datenblatt deiner
    SOLARPANELS - nicht auf dem Wechselrichter-Typenschild!""",
            "en": """The system power in Kilowatt-Peak (kWp) is the INSTALLED
power of your SOLAR PANELS - NOT the inverter!

  ┌─────────────────────────────────────────────────────────────────────┐
  │  kWp = Kilowatt-Peak = maximum power under standard conditions      │
  │                                                                     │
  │  Standard Test Conditions (STC):                                    │
  │    • 1000 W/m² solar irradiance                                     │
  │    • 25°C cell temperature                                          │
  │    • AM 1.5 spectrum                                                │
  └─────────────────────────────────────────────────────────────────────┘

⚠️  IMPORTANT: Find kWp on your SOLAR PANEL datasheet -
    not on the inverter nameplate!"""
        },
        "why": {
            "de": """Die KI nutzt die kWp-Angabe um zu berechnen, wieviel Energie
deine Anlage maximal produzieren KANN.

Damit kann sie:
  • Die theoretische Maximalproduktion berechnen
  • Die aktuelle Effizienz (Performance Ratio) ermitteln
  • Verschattung und Verschmutzung erkennen
  • Realistische Prognosen erstellen""",
            "en": """The AI uses the kWp value to calculate how much energy
your system CAN produce at maximum.

This allows it to:
  • Calculate theoretical maximum production
  • Determine current efficiency (Performance Ratio)
  • Detect shading and soiling
  • Create realistic forecasts"""
        },
        "unit": {"de": "Kilowatt-Peak (kWp)", "en": "Kilowatt-Peak (kWp)"},
        "entities": {
            "de": """Dies ist KEIN Sensor aus Home Assistant!

Du musst die kWp deiner Anlage MANUELL eingeben.
Die Information findest du:

  📄 Auf dem Datenblatt deiner Solarmodule
  📄 Im Angebot/Rechnung deines Installateurs
  📄 Im Einspeisevertrag mit dem Netzbetreiber
  📄 Auf dem Typenschild der Module (Wp pro Modul)""",
            "en": """This is NOT a sensor from Home Assistant!

You need to enter your system's kWp MANUALLY.
Find this information:

  📄 On your solar module datasheet
  📄 In the quote/invoice from your installer
  📄 In the feed-in contract with grid operator
  📄 On the module nameplate (Wp per module)"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Wechselrichter-Leistung statt Panel-Leistung
   FALSCH: "Mein WR hat 10 kW" → Das ist die WR-Kapazität!
   RICHTIG: Summe aller Solarpanel-Leistungen

❌ FEHLER 2: Verwechslung kW und kWp
   kW = aktuelle Leistung (variabel)
   kWp = installierte Maximalleistung (fest)

❌ FEHLER 3: Wp und kWp verwechselt
   Ein Panel hat z.B. 400 Wp = 0.4 kWp
   10 Panels × 400 Wp = 4000 Wp = 4.0 kWp

❌ FEHLER 4: Nur eine Dachseite gezählt
   Bei Ost-West-Anlage: BEIDE Seiten addieren!""",
            "en": """❌ MISTAKE 1: Inverter power instead of panel power
   WRONG: "My inverter has 10 kW" → That's inverter capacity!
   RIGHT: Sum of all solar panel powers

❌ MISTAKE 2: Confusing kW and kWp
   kW = current power (variable)
   kWp = installed peak power (fixed)

❌ MISTAKE 3: Confusing Wp and kWp
   One panel has e.g. 400 Wp = 0.4 kWp
   10 panels × 400 Wp = 4000 Wp = 4.0 kWp

❌ MISTAKE 4: Only counting one roof side
   For East-West system: ADD BOTH sides!"""
        },
        "tips": {
            "de": """💡 BERECHNUNG: Anzahl Module × Wp pro Modul ÷ 1000 = kWp

   Beispiel: 25 Module × 400 Wp = 10.000 Wp = 10,0 kWp

💡 Die kWp ist IMMER höher als die reale Produktion
   (wegen Temperatur, Winkel, Wolken, Verluste)

💡 Typische Jahresproduktion in Deutschland: ca. 950 kWh pro kWp
   → 10 kWp Anlage ≈ 9.500 kWh/Jahr

💡 Überdimensionierung: kWp > WR-Leistung ist normal und sinnvoll!""",
            "en": """💡 CALCULATION: Number of modules × Wp per module ÷ 1000 = kWp

   Example: 25 modules × 400 Wp = 10,000 Wp = 10.0 kWp

💡 kWp is ALWAYS higher than real production
   (due to temperature, angle, clouds, losses)

💡 Typical annual production in Germany: approx. 950 kWh per kWp
   → 10 kWp system ≈ 9,500 kWh/year

💡 Oversizing: kWp > inverter power is normal and useful!"""
        },
        "example": {
            "de": """BEISPIEL-BERECHNUNG:

┌─────────────────────────────────────────────────────────────────────┐
│  Dachseite Süd:    15 Module × 410 Wp = 6.150 Wp                   │
│  Dachseite Ost:     8 Module × 410 Wp = 3.280 Wp                   │
│  ────────────────────────────────────────────────────────────────  │
│  GESAMT:           23 Module × 410 Wp = 9.430 Wp = 9,43 kWp        │
└─────────────────────────────────────────────────────────────────────┘

Bei der Einrichtung von Solar Forecast gibst du ein:

  Gesamtleistung: 9.43 kWp

Oder bei Panel-Gruppen:
  Gruppe 1 (Süd): 6.15 kWp / 180° / 35°
  Gruppe 2 (Ost): 3.28 kWp / 90° / 25°""",
            "en": """EXAMPLE CALCULATION:

┌─────────────────────────────────────────────────────────────────────┐
│  Roof side South:  15 modules × 410 Wp = 6,150 Wp                  │
│  Roof side East:    8 modules × 410 Wp = 3,280 Wp                  │
│  ────────────────────────────────────────────────────────────────  │
│  TOTAL:            23 modules × 410 Wp = 9,430 Wp = 9.43 kWp       │
└─────────────────────────────────────────────────────────────────────┘

When setting up Solar Forecast, enter:

  Total power: 9.43 kWp

Or with panel groups:
  Group 1 (South): 6.15 kWp / 180° / 35°
  Group 2 (East): 3.28 kWp / 90° / 25°"""
        }
    },
    "power_sensor": {
        "name": {"de": "Power Sensor (DC-Leistung)", "en": "Power Sensor (DC Power)"},
        "mandatory": True,
        "show_dc_warning": True,
        "what": {
            "de": """Dein Power Sensor ist das Herzstück der Solar Forecast KI!

Er zeigt die AKTUELLE DC-Leistung deiner Solaranlage in Watt an.
Der Wert ändert sich ständig - von 0W nachts bis zum Maximum bei
strahlendem Sonnenschein.

⚡ ZWINGEND ERFORDERLICH: DC-Leistung (Gleichstrom)!

  DC-Leistung = Leistung direkt von den Solarpanels (VOR dem Wechselrichter)
  AC-Leistung = Leistung am Ausgang des Wechselrichters (FALSCH!)

Warum DC? Die KI muss wissen, was deine Panels wirklich produzieren.
Die AC-Leistung ist durch Wechselrichterverluste verfälscht (~3-5% weniger).""",
            "en": """Your Power Sensor is the heart of the Solar Forecast AI!

It shows the CURRENT DC power output of your solar system in Watts.
The value constantly changes - from 0W at night to maximum during
bright sunshine.

⚡ MANDATORY: DC power (direct current)!

  DC power = Power directly from solar panels (BEFORE the inverter)
  AC power = Power at inverter output (WRONG!)

Why DC? The AI needs to know what your panels really produce.
AC power is distorted by inverter losses (~3-5% less)."""
        },
        "why": {
            "de": """Ohne diesen Sensor kann die KI nicht lernen!

Er ist die Grundlage für alle Vorhersagen. Je länger die KI
deine Anlage beobachtet, desto genauer werden die Prognosen.

Die DC-Leistung zeigt das ECHTE Potential deiner Panels - unverfälscht
durch Wechselrichter-Verluste oder Batterie-Ladung.""",
            "en": """Without this sensor, the AI cannot learn!

It's the foundation for all predictions. The longer the AI
observes your system, the more accurate the forecasts become.

DC power shows the REAL potential of your panels - undistorted
by inverter losses or battery charging."""
        },
        "unit": {"de": "Watt (W) - DC/Gleichstrom", "en": "Watt (W) - DC/Direct Current"},
        "entities": {
            "de": """Fronius:       sensor.fronius_dc_power (NICHT ac_power!)
SMA:           sensor.sma_pv_power / sensor.sma_dc_power
Huawei:        sensor.inverter_input_power (DC-Eingang)
Kostal:        sensor.kostal_dc_power / sensor.kostal_pv_power
SolarEdge:     sensor.solaredge_dc_power
Growatt:       sensor.growatt_pv_power / sensor.growatt_pv1_power
Enphase:       sensor.envoy_current_power_production
Hoymiles:      sensor.hoymiles_pv_power / sensor.hoymiles_dc_power
APsystems:     sensor.apsystems_total_power
Deye:          sensor.deye_pv_power / sensor.deye_dc_power
Victron:       sensor.victron_pv_power

⚠️  Suche nach: "dc", "pv", "solar", "panel" im Sensornamen
⚠️  Vermeide: "ac", "output", "grid", "export" im Sensornamen""",
            "en": """Fronius:       sensor.fronius_dc_power (NOT ac_power!)
SMA:           sensor.sma_pv_power / sensor.sma_dc_power
Huawei:        sensor.inverter_input_power (DC input)
Kostal:        sensor.kostal_dc_power / sensor.kostal_pv_power
SolarEdge:     sensor.solaredge_dc_power
Growatt:       sensor.growatt_pv_power / sensor.growatt_pv1_power
Enphase:       sensor.envoy_current_power_production
Hoymiles:      sensor.hoymiles_pv_power / sensor.hoymiles_dc_power
APsystems:     sensor.apsystems_total_power
Deye:          sensor.deye_pv_power / sensor.deye_dc_power
Victron:       sensor.victron_pv_power

⚠️  Look for: "dc", "pv", "solar", "panel" in sensor name
⚠️  Avoid: "ac", "output", "grid", "export" in sensor name"""
        },
        "errors": {
            "de": """❌ FEHLER 1: AC-Leistung statt DC-Leistung!
   FALSCH: sensor.inverter_ac_power (AC = Wechselstrom-Ausgang!)
   FALSCH: sensor.grid_export_power (das ist Netzeinspeisung!)
   RICHTIG: sensor.inverter_dc_power / sensor.pv_power (DC = Gleichstrom)

❌ FEHLER 2: Falscher Sensor gewählt
   FALSCH: sensor.solar_energy_today (das ist Energie, nicht Leistung!)
   RICHTIG: sensor.solar_power (aktuelle Leistung in Watt)

❌ FEHLER 3: Sensor zeigt kW statt W
   Manche Wechselrichter liefern Kilowatt statt Watt.
   Prüfe: Zeigt der Sensor bei Sonne 5.2 oder 5200?
   Bei 5.2 → Template-Sensor erstellen (× 1000)

❌ FEHLER 4: Sensor ist immer 0 oder "unavailable"
   → Prüfe ob dein Wechselrichter online und erreichbar ist
   → Kontrolliere die Integration deines Wechselrichters
   → Warte auf Sonnenschein und prüfe dann erneut""",
            "en": """❌ MISTAKE 1: AC power instead of DC power!
   WRONG: sensor.inverter_ac_power (AC = alternating current output!)
   WRONG: sensor.grid_export_power (that's grid feed-in!)
   RIGHT: sensor.inverter_dc_power / sensor.pv_power (DC = direct current)

❌ MISTAKE 2: Wrong sensor selected
   WRONG: sensor.solar_energy_today (that's energy, not power!)
   RIGHT: sensor.solar_power (current power in Watts)

❌ MISTAKE 3: Sensor shows kW instead of W
   Some inverters provide Kilowatts instead of Watts.
   Check: Does the sensor show 5.2 or 5200 during sunshine?
   If 5.2 → Create template sensor (× 1000)

❌ MISTAKE 4: Sensor is always 0 or "unavailable"
   → Check if your inverter is online and reachable
   → Verify your inverter integration
   → Wait for sunshine and check again"""
        },
        "tips": {
            "de": """💡 DC-Leistung erkennst du oft an "DC", "PV" oder "Panel" im Namen
💡 Der Wert sollte bei Sonnenschein mehrere hundert bis tausend Watt zeigen
💡 Nachts sollte der Wert 0 oder nahe 0 sein
💡 Schwankungen bei Wolken sind völlig normal
💡 Öffne Entwicklerwerkzeuge → Zustände und beobachte den Sensor live""",
            "en": """💡 DC power often has "DC", "PV" or "Panel" in the sensor name
💡 The value should show several hundred to thousand Watts during sunshine
💡 At night, the value should be 0 or close to 0
💡 Fluctuations during clouds are completely normal
💡 Open Developer Tools → States and watch the sensor live"""
        },
        "example": {
            "de": """Falls dein Wechselrichter kW statt W liefert:

template:
  - sensor:
      - name: "Solar DC Power Watts"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: "{{ states('sensor.inverter_dc_power_kw') | float * 1000 }}"
        availability: "{{ states('sensor.inverter_dc_power_kw') not in ['unknown', 'unavailable'] }}"

Falls du mehrere MPPT-Tracker hast, addiere sie:

template:
  - sensor:
      - name: "Total DC Power"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {{ (states('sensor.pv1_power') | float(0)) +
             (states('sensor.pv2_power') | float(0)) }}""",
            "en": """If your inverter provides kW instead of W:

template:
  - sensor:
      - name: "Solar DC Power Watts"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: "{{ states('sensor.inverter_dc_power_kw') | float * 1000 }}"
        availability: "{{ states('sensor.inverter_dc_power_kw') not in ['unknown', 'unavailable'] }}"

If you have multiple MPPT trackers, add them:

template:
  - sensor:
      - name: "Total DC Power"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {{ (states('sensor.pv1_power') | float(0)) +
             (states('sensor.pv2_power') | float(0)) }}"""
        }
    },
    "daily_yield": {
        "name": {"de": "Daily Yield (DC-Tagesertrag)", "en": "Daily Yield (DC Production)"},
        "mandatory": True,
        "show_dc_warning": True,
        "what": {
            "de": """Der Daily Yield Sensor zeigt die HEUTE produzierte DC-Energie in kWh.

⚡ ZWINGEND ERFORDERLICH: DC-Energie (Gleichstrom)!

  DC-Energie = Energie direkt von den Solarpanels (VOR dem Wechselrichter)
  AC-Energie = Energie am Ausgang des Wechselrichters (FALSCH!)

Er akkumuliert über den Tag:
  Morgens:  0 kWh
  Mittags:  ~15 kWh
  Abends:   ~25 kWh (je nach Anlage und Wetter)

⚠️  WICHTIG: Der Sensor MUSS jeden Tag um Mitternacht auf 0 zurücksetzen!""",
            "en": """The Daily Yield Sensor shows the DC energy produced TODAY in kWh.

⚡ MANDATORY: DC energy (direct current)!

  DC energy = Energy directly from solar panels (BEFORE the inverter)
  AC energy = Energy at inverter output (WRONG!)

It accumulates throughout the day:
  Morning:  0 kWh
  Noon:     ~15 kWh
  Evening:  ~25 kWh (depending on system and weather)

⚠️  IMPORTANT: The sensor MUST reset to 0 every day at midnight!"""
        },
        "why": {
            "de": """Die KI nutzt diesen Sensor um zu lernen, wieviel Energie deine
Anlage an verschiedenen Tagen produziert.

Der tägliche Reset ist essentiell - sonst kann die KI nicht
zwischen einzelnen Tagen unterscheiden!

DC-Energie zeigt die ECHTE Produktion deiner Panels - ohne
Wechselrichter-Verluste oder Eigenverbrauch-Abzüge.""",
            "en": """The AI uses this sensor to learn how much energy your system
produces on different days.

The daily reset is essential - otherwise the AI cannot
distinguish between individual days!

DC energy shows the REAL production of your panels - without
inverter losses or self-consumption deductions."""
        },
        "unit": {"de": "Kilowattstunden (kWh) - DC/Gleichstrom", "en": "Kilowatt-hours (kWh) - DC/Direct Current"},
        "entities": {
            "de": """Fronius:       sensor.fronius_energy_day (DC, nicht AC!)
SMA:           sensor.sma_pv_gen_meter / sensor.sma_daily_yield
Huawei:        sensor.inverter_daily_yield (prüfe ob DC!)
Kostal:        sensor.kostal_daily_energy / sensor.kostal_home_own_consumption_from_pv
SolarEdge:     sensor.solaredge_energy_today
Growatt:       sensor.growatt_today_generate_energy
Enphase:       sensor.envoy_today_energy_production
Hoymiles:      sensor.hoymiles_today_production
APsystems:     sensor.apsystems_today_energy
Deye:          sensor.deye_daily_production

⚠️  Suche nach: "pv", "solar", "yield", "production" im Sensornamen
⚠️  Vermeide: "ac", "grid", "export", "feed" im Sensornamen""",
            "en": """Fronius:       sensor.fronius_energy_day (DC, not AC!)
SMA:           sensor.sma_pv_gen_meter / sensor.sma_daily_yield
Huawei:        sensor.inverter_daily_yield (check if DC!)
Kostal:        sensor.kostal_daily_energy / sensor.kostal_home_own_consumption_from_pv
SolarEdge:     sensor.solaredge_energy_today
Growatt:       sensor.growatt_today_generate_energy
Enphase:       sensor.envoy_today_energy_production
Hoymiles:      sensor.hoymiles_today_production
APsystems:     sensor.apsystems_today_energy
Deye:          sensor.deye_daily_production

⚠️  Look for: "pv", "solar", "yield", "production" in sensor name
⚠️  Avoid: "ac", "grid", "export", "feed" in sensor name"""
        },
        "errors": {
            "de": """❌ FEHLER 1: AC-Energie statt DC-Energie!
   FALSCH: sensor.grid_feed_in_today (das ist Netzeinspeisung!)
   FALSCH: sensor.inverter_ac_energy_today (AC = nach Wechselrichter)
   RICHTIG: sensor.pv_energy_today / sensor.dc_energy_today

❌ FEHLER 2: Sensor setzt NICHT auf 0 zurück
   PROBLEM: Sensor zeigt z.B. 15000 kWh (Gesamtertrag seit Installation)
   LÖSUNG: Du brauchst einen Utility Meter Helper (siehe Beispiel)

❌ FEHLER 3: Leistung statt Energie gewählt
   FALSCH: sensor.solar_power (Leistung in Watt!)
   RICHTIG: sensor.solar_energy_today (Energie in kWh)

❌ FEHLER 4: Einheit ist Wh statt kWh
   Manche Wechselrichter liefern Wattstunden statt Kilowattstunden.
   Falls der Sensor 25000 statt 25 zeigt → Template-Sensor (÷ 1000)

❌ FEHLER 5: Reset zu spät oder zu früh
   Der Sensor muss exakt um 00:00 Uhr zurücksetzen.
   Falls nicht → Utility Meter mit cycle: daily nutzen""",
            "en": """❌ MISTAKE 1: AC energy instead of DC energy!
   WRONG: sensor.grid_feed_in_today (that's grid export!)
   WRONG: sensor.inverter_ac_energy_today (AC = after inverter)
   RIGHT: sensor.pv_energy_today / sensor.dc_energy_today

❌ MISTAKE 2: Sensor does NOT reset to 0
   PROBLEM: Sensor shows e.g. 15000 kWh (total yield since installation)
   SOLUTION: You need a Utility Meter Helper (see example)

❌ MISTAKE 3: Power instead of energy selected
   WRONG: sensor.solar_power (power in Watts!)
   RIGHT: sensor.solar_energy_today (energy in kWh)

❌ MISTAKE 4: Unit is Wh instead of kWh
   Some inverters provide Watt-hours instead of Kilowatt-hours.
   If sensor shows 25000 instead of 25 → Template sensor (÷ 1000)

❌ MISTAKE 5: Reset too late or too early
   The sensor must reset exactly at 00:00.
   If not → Use Utility Meter with cycle: daily"""
        },
        "tips": {
            "de": """💡 DC-Energie ist die Energie VOR dem Wechselrichter (vom Dach)
💡 Prüfe um 00:05 Uhr ob der Sensor auf 0 steht
💡 Vergleiche den Abendwert mit deiner Wechselrichter-App
💡 Der Wert sollte NIEMALS negativ sein
💡 Nach HA-Neustart sollte der Wert erhalten bleiben""",
            "en": """💡 DC energy is the energy BEFORE the inverter (from the roof)
💡 Check at 00:05 if the sensor is at 0
💡 Compare the evening value with your inverter app
💡 The value should NEVER be negative
💡 After HA restart, the value should persist"""
        },
        "example": {
            "de": """Utility Meter für täglichen Reset (in configuration.yaml):

utility_meter:
  solar_daily_yield:
    source: sensor.solar_total_dc_energy
    cycle: daily
    name: "Solar DC Tagesertrag"

Wichtig: 'source' muss ein Sensor mit stetig steigendem DC-Wert sein!

─────────────────────────────────────────────────────────────

Falls dein Sensor Wh statt kWh liefert:

template:
  - sensor:
      - name: "Solar Daily Yield kWh"
        unit_of_measurement: "kWh"
        device_class: energy
        state_class: total_increasing
        state: "{{ states('sensor.solar_daily_wh') | float / 1000 }}" """,
            "en": """Utility Meter for daily reset (in configuration.yaml):

utility_meter:
  solar_daily_yield:
    source: sensor.solar_total_dc_energy
    cycle: daily
    name: "Solar DC Daily Yield"

Important: 'source' must be a sensor with continuously increasing DC value!

─────────────────────────────────────────────────────────────

If your sensor provides Wh instead of kWh:

template:
  - sensor:
      - name: "Solar Daily Yield kWh"
        unit_of_measurement: "kWh"
        device_class: energy
        state_class: total_increasing
        state: "{{ states('sensor.solar_daily_wh') | float / 1000 }}" """
        }
    },
    "temperature": {
        "name": {"de": "Temperatur Sensor", "en": "Temperature Sensor"},
        "mandatory": False,
        "what": {
            "de": """Ein LOKALER Außentemperatur-Sensor an deinem Standort.

Die Temperatur beeinflusst die Effizienz deiner Solarpanels erheblich:
  • Kälter = höhere Effizienz (Panels mögen Kälte!)
  • Wärmer = niedrigere Effizienz (ca. -0.4% pro °C über 25°C)

An einem heißen Sommertag kann die Produktion allein durch die
Temperatur um 10-15% niedriger sein als an einem kühlen Tag!""",
            "en": """A LOCAL outdoor temperature sensor at your location.

Temperature significantly affects your solar panel efficiency:
  • Colder = higher efficiency (panels love cold!)
  • Warmer = lower efficiency (approx. -0.4% per °C above 25°C)

On a hot summer day, production can be 10-15% lower than on
a cool day just because of temperature!"""
        },
        "why": {
            "de": """Die KI kann mit diesem Sensor den Temperatur-Effekt auf deine
Panels verstehen und in die Vorhersage einbeziehen.

Besonders im Sommer macht das einen großen Unterschied!""",
            "en": """With this sensor, the AI can understand the temperature effect
on your panels and include it in predictions.

This makes a big difference, especially in summer!"""
        },
        "unit": {"de": "Grad Celsius (°C)", "en": "Degrees Celsius (°C)"},
        "entities": {
            "de": """Netatmo:       sensor.netatmo_outdoor_temperature
Ecowitt:       sensor.ecowitt_outdoor_temperature
Bresser:       sensor.bresser_outdoor_temp
TFA Dostmann:  sensor.tfa_outdoor_temperature
Aqara:         sensor.aqara_weather_temperature
Shelly H&T:    sensor.shelly_ht_temperature (outdoor montiert)
Homematic:     sensor.hmip_outdoor_temperature
Zigbee:        sensor.zigbee_outdoor_temp_sensor""",
            "en": """Netatmo:       sensor.netatmo_outdoor_temperature
Ecowitt:       sensor.ecowitt_outdoor_temperature
Bresser:       sensor.bresser_outdoor_temp
TFA Dostmann:  sensor.tfa_outdoor_temperature
Aqara:         sensor.aqara_weather_temperature
Shelly H&T:    sensor.shelly_ht_temperature (mounted outdoor)
Homematic:     sensor.hmip_outdoor_temperature
Zigbee:        sensor.zigbee_outdoor_temp_sensor"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Innentemperatur statt Außentemperatur
   FALSCH: sensor.living_room_temperature (das ist drinnen!)
   RICHTIG: sensor.outdoor_temperature (draußen!)

❌ FEHLER 2: Wetter-App statt lokalem Sensor
   FALSCH: sensor.openweathermap_temperature (das sind keine lokalen Daten!)
   RICHTIG: Nur echte Hardware-Sensoren an deinem Standort!

❌ FEHLER 3: Sensor in Fahrenheit
   Falls dein Sensor Fahrenheit liefert, erstelle einen Template-Sensor:
   {{ (states('sensor.temp_f') | float - 32) * 5/9 }}""",
            "en": """❌ MISTAKE 1: Indoor temperature instead of outdoor
   WRONG: sensor.living_room_temperature (that's inside!)
   RIGHT: sensor.outdoor_temperature (outside!)

❌ MISTAKE 2: Weather app instead of local sensor
   WRONG: sensor.openweathermap_temperature (that's not local data!)
   RIGHT: Only real hardware sensors at your location!

❌ MISTAKE 3: Sensor in Fahrenheit
   If your sensor provides Fahrenheit, create a template sensor:
   {{ (states('sensor.temp_f') | float - 32) * 5/9 }}"""
        },
        "tips": {
            "de": """💡 Platziere den Sensor im Schatten, nicht in der Sonne
💡 Günstige Zigbee/WiFi Sensoren (Aqara, Shelly) funktionieren gut
💡 Der Sensor sollte die echte Lufttemperatur zeigen, nicht "gefühlt" """,
            "en": """💡 Place the sensor in shade, not in direct sunlight
💡 Cheap Zigbee/WiFi sensors (Aqara, Shelly) work well
💡 The sensor should show real air temperature, not "feels like" """
        },
        "example": {
            "de": """Kein Template nötig - verwende den Sensor direkt!

Falls Fahrenheit → Celsius Umrechnung nötig:

template:
  - sensor:
      - name: "Außentemperatur Celsius"
        unit_of_measurement: "°C"
        device_class: temperature
        state_class: measurement
        state: "{{ ((states('sensor.outdoor_temp_f') | float) - 32) * 5/9 | round(1) }}" """,
            "en": """No template needed - use the sensor directly!

If Fahrenheit → Celsius conversion needed:

template:
  - sensor:
      - name: "Outdoor Temperature Celsius"
        unit_of_measurement: "°C"
        device_class: temperature
        state_class: measurement
        state: "{{ ((states('sensor.outdoor_temp_f') | float) - 32) * 5/9 | round(1) }}" """
        }
    },
    "lux": {
        "name": {"de": "Lichtstärke (Lux)", "en": "Illuminance (Lux)"},
        "mandatory": False,
        "what": {
            "de": """Ein LOKALER Lux-Sensor misst die Helligkeit am Standort.

Lux korreliert direkt mit der möglichen Solarproduktion:
  • 0 lux = Nacht
  • 10.000 lux = Bedeckter Himmel
  • 50.000 lux = Leicht bewölkt
  • 100.000+ lux = Volle Sonne

Das ist einer der wertvollsten optionalen Sensoren!""",
            "en": """A LOCAL lux sensor measures brightness at your location.

Lux correlates directly with possible solar production:
  • 0 lux = Night
  • 10,000 lux = Overcast sky
  • 50,000 lux = Partly cloudy
  • 100,000+ lux = Full sun

This is one of the most valuable optional sensors!"""
        },
        "why": {
            "de": """Die KI kann mit einem Lux-Sensor die aktuelle
Lichtsituation direkt messen - viel besser als Wetterdaten!

Ein lokaler Lux-Sensor "sieht" auch lokale Wolken,
Nebel oder Schatten.""",
            "en": """With a lux sensor, the AI can directly measure the
current light situation - much better than weather data!

A local lux sensor also "sees" local clouds,
fog, or shadows."""
        },
        "unit": {"de": "Lux (lx)", "en": "Lux (lx)"},
        "entities": {
            "de": """Xiaomi/Aqara:  sensor.aqara_illuminance
Philips Hue:   sensor.hue_outdoor_motion_illuminance
Ecowitt:       sensor.ecowitt_solar_lux
Homematic:     sensor.hmip_illuminance
Fibaro:        sensor.fibaro_motion_illuminance
Zigbee:        sensor.zigbee_lux_sensor
ESP/DIY:       sensor.bh1750_illuminance""",
            "en": """Xiaomi/Aqara:  sensor.aqara_illuminance
Philips Hue:   sensor.hue_outdoor_motion_illuminance
Ecowitt:       sensor.ecowitt_solar_lux
Homematic:     sensor.hmip_illuminance
Fibaro:        sensor.fibaro_motion_illuminance
Zigbee:        sensor.zigbee_lux_sensor
ESP/DIY:       sensor.bh1750_illuminance"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Indoor-Sensor statt Outdoor
   Der Sensor MUSS draußen oder am Fenster mit freier Sicht sein!

❌ FEHLER 2: Sensor im Schatten platziert
   Der Sensor sollte möglichst viel Himmel "sehen".""",
            "en": """❌ MISTAKE 1: Indoor sensor instead of outdoor
   The sensor MUST be outside or at a window with clear sky view!

❌ MISTAKE 2: Sensor placed in shade
   The sensor should "see" as much sky as possible."""
        },
        "tips": {
            "de": """💡 Wenn möglich: Solar Radiation (W/m²) ist noch besser als Lux
💡 Günstige Zigbee-Sensoren (Aqara, Xiaomi) funktionieren gut
💡 Platziere den Sensor mit freier Sicht zum Himmel""",
            "en": """💡 If possible: Solar Radiation (W/m²) is even better than Lux
💡 Cheap Zigbee sensors (Aqara, Xiaomi) work well
💡 Place the sensor with clear view to the sky"""
        },
        "example": {
            "de": """Kein Template nötig - verwende den Sensor direkt!""",
            "en": """No template needed - use the sensor directly!"""
        }
    },
    "solar_radiation": {
        "name": {"de": "Solarstrahlung (W/m²)", "en": "Solar Radiation (W/m²)"},
        "mandatory": False,
        "what": {
            "de": """Ein LOKALER Solar Radiation Sensor - der BESTE optionale Sensor!

Er misst direkt, wieviel Sonnenenergie auf eine Fläche trifft:
  • 0 W/m² = Nacht
  • 200-400 W/m² = Bedeckt
  • 600-800 W/m² = Leicht bewölkt
  • 800-1000 W/m² = Volle Sonne

Besser als Lux, da direkt in Energie-Einheiten gemessen wird!""",
            "en": """A LOCAL Solar Radiation Sensor - the BEST optional sensor!

It directly measures how much solar energy hits a surface:
  • 0 W/m² = Night
  • 200-400 W/m² = Overcast
  • 600-800 W/m² = Partly cloudy
  • 800-1000 W/m² = Full sun

Better than Lux because it's measured directly in energy units!"""
        },
        "why": {
            "de": """Dieser Sensor misst GENAU das, was deine Panels empfangen!

Die KI kann damit extrem genaue Vorhersagen treffen.""",
            "en": """This sensor measures EXACTLY what your panels receive!

The AI can make extremely accurate predictions with this."""
        },
        "unit": {"de": "Watt pro Quadratmeter (W/m²)", "en": "Watts per square meter (W/m²)"},
        "entities": {
            "de": """Ecowitt:       sensor.ecowitt_solar_radiation
Davis:         sensor.davis_solar_radiation
Bresser:       sensor.bresser_uv_solar_radiation
ESP/DIY:       sensor.esp_solar_radiation""",
            "en": """Ecowitt:       sensor.ecowitt_solar_radiation
Davis:         sensor.davis_solar_radiation
Bresser:       sensor.bresser_uv_solar_radiation
ESP/DIY:       sensor.esp_solar_radiation"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Sensor muss HORIZONTAL montiert sein
❌ FEHLER 2: Sensor verschmutzt - regelmäßig reinigen""",
            "en": """❌ MISTAKE 1: Sensor must be mounted HORIZONTALLY
❌ MISTAKE 2: Sensor dirty - clean regularly"""
        },
        "tips": {
            "de": """💡 DER beste optionale Sensor für Solar Forecast!
💡 Ecowitt Wetterstationen haben diesen Sensor oft eingebaut
💡 Priorität: Solar Radiation > Lux > kein Lichtsensor""",
            "en": """💡 THE best optional sensor for Solar Forecast!
💡 Ecowitt weather stations often have this sensor built in
💡 Priority: Solar Radiation > Lux > no light sensor"""
        },
        "example": {
            "de": """Kein Template nötig - verwende den Sensor direkt!""",
            "en": """No template needed - use the sensor directly!"""
        }
    },
    "solar_to_battery": {
        "name": {"de": "Solar → Batterie", "en": "Solar → Battery"},
        "mandatory": False,
        "show_battery_info": True,
        "what": {
            "de": """Dieser Sensor zeigt, wieviel Solarstrom DIREKT in die Batterie fließt.

⚡ WICHTIG für Zero-Export-Anlagen!

Bei Zero-Export wird überschüssiger Strom nicht ins Netz eingespeist,
sondern gedrosselt. Der Power-Sensor zeigt dann NICHT die echte
potentielle Produktion!

Mit diesem Sensor kann die KI berechnen:
  Echte Produktion = Power + Solar-zu-Batterie

So lernt die KI das WAHRE Potential deiner Anlage!""",
            "en": """This sensor shows how much solar power flows DIRECTLY into the battery.

⚡ IMPORTANT for Zero-Export systems!

In Zero-Export, excess power is not fed to the grid,
but throttled. The power sensor then does NOT show the real
potential production!

With this sensor, the AI can calculate:
  Real Production = Power + Solar-to-Battery

This way the AI learns the TRUE potential of your system!"""
        },
        "why": {
            "de": """Ohne diesen Sensor unterschätzt die KI bei Zero-Export-Anlagen
die tatsächliche Produktionskapazität erheblich!

Die KI sieht nur die gedrosselte Leistung und denkt, das sei
das Maximum. Mit Solar-zu-Batterie versteht sie das echte Potential.""",
            "en": """Without this sensor, the AI significantly underestimates
the actual production capacity of Zero-Export systems!

The AI only sees the throttled power and thinks that's the
maximum. With Solar-to-Battery it understands the real potential."""
        },
        "unit": {"de": "Watt (W)", "en": "Watt (W)"},
        "entities": {
            "de": """Fronius:       sensor.fronius_power_battery_charge
SMA:           sensor.sma_battery_charging_power
Huawei:        sensor.battery_charge_power
Kostal:        sensor.kostal_battery_charge_power
SolarEdge:     sensor.solaredge_battery_power (positiv = laden)
Growatt:       sensor.growatt_battery_charge_power
Victron:       sensor.victron_battery_power
BYD:           sensor.byd_battery_charge_power
Pylontech:     sensor.pylontech_charge_power""",
            "en": """Fronius:       sensor.fronius_power_battery_charge
SMA:           sensor.sma_battery_charging_power
Huawei:        sensor.battery_charge_power
Kostal:        sensor.kostal_battery_charge_power
SolarEdge:     sensor.solaredge_battery_power (positive = charging)
Growatt:       sensor.growatt_battery_charge_power
Victron:       sensor.victron_battery_power
BYD:           sensor.byd_battery_charge_power
Pylontech:     sensor.pylontech_charge_power"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Batterie-Entladung statt Ladung
   FALSCH: sensor.battery_discharge_power (das ist Entladen!)
   RICHTIG: sensor.battery_charge_power (Laden aus Solar)

❌ FEHLER 2: Gesamte Batterieleistung statt nur Solar
   Manche Sensoren zeigen Laden aus Solar + Netz zusammen.
   Du brauchst NUR den Anteil, der von Solar kommt!

❌ FEHLER 3: Negative Werte nicht beachtet
   Manche WR zeigen Laden als negativ, Entladen als positiv.
   → Template-Sensor mit abs() oder Vorzeichen-Logik erstellen""",
            "en": """❌ MISTAKE 1: Battery discharge instead of charge
   WRONG: sensor.battery_discharge_power (that's discharging!)
   RIGHT: sensor.battery_charge_power (charging from solar)

❌ MISTAKE 2: Total battery power instead of solar only
   Some sensors show charging from solar + grid combined.
   You need ONLY the part that comes from solar!

❌ MISTAKE 3: Not handling negative values
   Some inverters show charging as negative, discharging as positive.
   → Create template sensor with abs() or sign logic"""
        },
        "tips": {
            "de": """💡 Nur relevant wenn du eine Batterie UND Zero-Export hast
💡 Bei Einspeisung ins Netz ist dieser Sensor weniger wichtig
💡 Der Wert sollte bei Sonnenschein und voller Batterie = 0 sein
💡 Prüfe: Steigt der Wert wenn die Batterie aus Solar lädt?""",
            "en": """💡 Only relevant if you have a battery AND Zero-Export
💡 With grid feed-in, this sensor is less important
💡 Value should be 0 during sunshine when battery is full
💡 Check: Does value increase when battery charges from solar?"""
        },
        "example": {
            "de": """Falls dein Sensor negative Werte für Laden zeigt:

template:
  - sensor:
      - name: "Solar zu Batterie"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {% set val = states('sensor.battery_power') | float(0) %}
          {{ val | abs if val < 0 else 0 }}

Falls dein Sensor Laden/Entladen kombiniert:

template:
  - sensor:
      - name: "Solar zu Batterie"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {% set val = states('sensor.battery_power') | float(0) %}
          {{ val if val > 0 else 0 }}""",
            "en": """If your sensor shows negative values for charging:

template:
  - sensor:
      - name: "Solar to Battery"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {% set val = states('sensor.battery_power') | float(0) %}
          {{ val | abs if val < 0 else 0 }}

If your sensor combines charging/discharging:

template:
  - sensor:
      - name: "Solar to Battery"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {% set val = states('sensor.battery_power') | float(0) %}
          {{ val if val > 0 else 0 }}"""
        }
    },
    "consumption": {
        "name": {"de": "Hausverbrauch", "en": "House Consumption"},
        "mandatory": False,
        "what": {
            "de": """Der Hausverbrauch-Sensor zeigt den aktuellen Stromverbrauch deines Haushalts.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Solarproduktion = Eigenverbrauch + Einspeisung + Batterieladen │
  └─────────────────────────────────────────────────────────────────┘

Mit diesem Sensor kann die KI verstehen, wie dein Verbrauchsmuster
aussieht und bessere Empfehlungen geben.""",
            "en": """The house consumption sensor shows the current power consumption of your household.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Solar Production = Self-consumption + Grid Export + Battery    │
  └─────────────────────────────────────────────────────────────────┘

With this sensor, the AI can understand your consumption patterns
and provide better recommendations."""
        },
        "why": {
            "de": """Die KI kann mit diesem Sensor:
  • Eigenverbrauchsquoten berechnen
  • Optimale Zeiten für Großverbraucher vorschlagen
  • Bessere Statistiken über Autarkie liefern""",
            "en": """With this sensor, the AI can:
  • Calculate self-consumption rates
  • Suggest optimal times for heavy loads
  • Provide better statistics on self-sufficiency"""
        },
        "unit": {"de": "Watt (W)", "en": "Watt (W)"},
        "entities": {
            "de": """Shelly EM/3EM: sensor.shelly_em_channel_power
Fronius:       sensor.fronius_house_load
SMA:           sensor.sma_house_consumption
Huawei:        sensor.house_consumption_power
Kostal:        sensor.kostal_home_consumption
SolarEdge:     sensor.solaredge_house_consumption
Growatt:       sensor.growatt_local_load
Victron:       sensor.victron_ac_consumption
Smartmeter:    sensor.smartmeter_power_consumption""",
            "en": """Shelly EM/3EM: sensor.shelly_em_channel_power
Fronius:       sensor.fronius_house_load
SMA:           sensor.sma_house_consumption
Huawei:        sensor.house_consumption_power
Kostal:        sensor.kostal_home_consumption
SolarEdge:     sensor.solaredge_house_consumption
Growatt:       sensor.growatt_local_load
Victron:       sensor.victron_ac_consumption
Smartmeter:    sensor.smartmeter_power_consumption"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Nur einzelne Phasen statt Gesamtverbrauch
   Bei 3-Phasen-Anschluss alle Phasen addieren!

❌ FEHLER 2: Netzexport statt Verbrauch
   FALSCH: sensor.grid_export_power
   RICHTIG: sensor.house_consumption""",
            "en": """❌ MISTAKE 1: Single phases instead of total consumption
   With 3-phase connection, add all phases!

❌ MISTAKE 2: Grid export instead of consumption
   WRONG: sensor.grid_export_power
   RIGHT: sensor.house_consumption"""
        },
        "tips": {
            "de": """💡 Shelly EM oder 3EM sind günstige und genaue Lösungen
💡 Der Verbrauch sollte nie negativ sein
💡 Typischer Grundverbrauch: 200-500W (Standby, Kühlschrank etc.)""",
            "en": """💡 Shelly EM or 3EM are affordable and accurate solutions
💡 Consumption should never be negative
💡 Typical base load: 200-500W (standby, fridge etc.)"""
        },
        "example": {
            "de": """Summe aus 3 Phasen (z.B. Shelly 3EM):

template:
  - sensor:
      - name: "Hausverbrauch Gesamt"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {{ (states('sensor.shelly_3em_channel_a_power') | float(0)) +
             (states('sensor.shelly_3em_channel_b_power') | float(0)) +
             (states('sensor.shelly_3em_channel_c_power') | float(0)) }}""",
            "en": """Sum of 3 phases (e.g., Shelly 3EM):

template:
  - sensor:
      - name: "Total House Consumption"
        unit_of_measurement: "W"
        device_class: power
        state_class: measurement
        state: >
          {{ (states('sensor.shelly_3em_channel_a_power') | float(0)) +
             (states('sensor.shelly_3em_channel_b_power') | float(0)) +
             (states('sensor.shelly_3em_channel_c_power') | float(0)) }}"""
        }
    },
    "humidity": {
        "name": {"de": "Luftfeuchtigkeit", "en": "Humidity"},
        "mandatory": False,
        "what": {
            "de": """Ein LOKALER Luftfeuchtigkeitssensor an deinem Standort.

Hohe Luftfeuchtigkeit kann auf:
  • Nebel und Dunst hindeuten
  • Erhöhte Wolkenbildung
  • Reduzierte Sonneneinstrahlung

Die KI nutzt diesen Wert als zusätzlichen Indikator für Wetterbedingungen.""",
            "en": """A LOCAL humidity sensor at your location.

High humidity can indicate:
  • Fog and haze
  • Increased cloud formation
  • Reduced solar irradiance

The AI uses this value as an additional indicator for weather conditions."""
        },
        "why": {
            "de": """Zusammen mit Temperatur hilft Luftfeuchtigkeit der KI,
lokale Wetterbedingungen besser zu verstehen - besonders bei
Nebel oder Dunst am Morgen.""",
            "en": """Together with temperature, humidity helps the AI
better understand local weather conditions - especially for
fog or haze in the morning."""
        },
        "unit": {"de": "Prozent (%)", "en": "Percent (%)"},
        "entities": {
            "de": """Netatmo:       sensor.netatmo_outdoor_humidity
Ecowitt:       sensor.ecowitt_outdoor_humidity
Aqara:         sensor.aqara_weather_humidity
Shelly H&T:    sensor.shelly_ht_humidity
Homematic:     sensor.hmip_outdoor_humidity
Zigbee:        sensor.zigbee_humidity_sensor""",
            "en": """Netatmo:       sensor.netatmo_outdoor_humidity
Ecowitt:       sensor.ecowitt_outdoor_humidity
Aqara:         sensor.aqara_weather_humidity
Shelly H&T:    sensor.shelly_ht_humidity
Homematic:     sensor.hmip_outdoor_humidity
Zigbee:        sensor.zigbee_humidity_sensor"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Innen- statt Außensensor
   FALSCH: sensor.bathroom_humidity
   RICHTIG: sensor.outdoor_humidity

❌ FEHLER 2: Wetter-App statt lokalem Sensor
   FALSCH: sensor.openweathermap_humidity
   RICHTIG: Nur echte Hardware-Sensoren!""",
            "en": """❌ MISTAKE 1: Indoor instead of outdoor sensor
   WRONG: sensor.bathroom_humidity
   RIGHT: sensor.outdoor_humidity

❌ MISTAKE 2: Weather app instead of local sensor
   WRONG: sensor.openweathermap_humidity
   RIGHT: Only real hardware sensors!"""
        },
        "tips": {
            "de": """💡 Meist zusammen mit Temperatursensor in einem Gerät
💡 Platziere den Sensor im Schatten, geschützt vor Regen
💡 Werte über 90% deuten oft auf Nebel hin""",
            "en": """💡 Usually combined with temperature sensor in one device
💡 Place sensor in shade, protected from rain
💡 Values above 90% often indicate fog"""
        },
        "example": {
            "de": """Kein Template nötig - verwende den Sensor direkt!""",
            "en": """No template needed - use the sensor directly!"""
        }
    },
    "wind": {
        "name": {"de": "Windgeschwindigkeit", "en": "Wind Speed"},
        "mandatory": False,
        "what": {
            "de": """Ein LOKALER Windmesser an deinem Standort.

Wind beeinflusst die Solarproduktion:
  • Kühlt die Panels → höhere Effizienz
  • Kann Wolken schnell vorbeiziehen lassen
  • Starker Wind kann bei manchen Anlagen zur Abschaltung führen

Windgeschwindigkeit hilft der KI, Temperatureffekte besser zu verstehen.""",
            "en": """A LOCAL wind meter at your location.

Wind affects solar production:
  • Cools panels → higher efficiency
  • Can move clouds quickly
  • Strong wind may cause shutdown on some systems

Wind speed helps the AI better understand temperature effects."""
        },
        "why": {
            "de": """Die KI kann mit diesem Sensor den Kühleffekt auf die Panels
besser einschätzen. An windigen Tagen können Panels trotz
hoher Temperaturen effizienter arbeiten.""",
            "en": """With this sensor, the AI can better estimate the cooling
effect on panels. On windy days, panels can work more
efficiently despite high temperatures."""
        },
        "unit": {"de": "Meter pro Sekunde (m/s) oder km/h", "en": "Meters per second (m/s) or km/h"},
        "entities": {
            "de": """Ecowitt:       sensor.ecowitt_wind_speed
Netatmo:       sensor.netatmo_wind_strength
Davis:         sensor.davis_wind_speed
Bresser:       sensor.bresser_wind_speed
Homematic:     sensor.hmip_wind_speed""",
            "en": """Ecowitt:       sensor.ecowitt_wind_speed
Netatmo:       sensor.netatmo_wind_strength
Davis:         sensor.davis_wind_speed
Bresser:       sensor.bresser_wind_speed
Homematic:     sensor.hmip_wind_speed"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Windböen statt Durchschnitt
   Böen sind kurze Spitzen - nutze den Durchschnittswert!

❌ FEHLER 2: Wetter-App statt lokalem Sensor
   FALSCH: sensor.openweathermap_wind
   RICHTIG: Nur echte Hardware-Sensoren!""",
            "en": """❌ MISTAKE 1: Wind gusts instead of average
   Gusts are short peaks - use the average value!

❌ MISTAKE 2: Weather app instead of local sensor
   WRONG: sensor.openweathermap_wind
   RIGHT: Only real hardware sensors!"""
        },
        "tips": {
            "de": """💡 Ideal: Wetterstation mit Windmesser auf dem Dach
💡 Windmesser sollte möglichst frei stehen (keine Gebäude davor)
💡 Weniger wichtig als Temperatur und Lux""",
            "en": """💡 Ideal: Weather station with wind meter on the roof
💡 Wind meter should be placed freely (no buildings in front)
💡 Less important than temperature and lux"""
        },
        "example": {
            "de": """Falls km/h → m/s Umrechnung nötig:

template:
  - sensor:
      - name: "Wind m/s"
        unit_of_measurement: "m/s"
        state_class: measurement
        state: "{{ (states('sensor.wind_kmh') | float(0)) / 3.6 | round(1) }}" """,
            "en": """If km/h → m/s conversion needed:

template:
  - sensor:
      - name: "Wind m/s"
        unit_of_measurement: "m/s"
        state_class: measurement
        state: "{{ (states('sensor.wind_kmh') | float(0)) / 3.6 | round(1) }}" """
        }
    },
    "rain": {
        "name": {"de": "Regensensor", "en": "Rain Sensor"},
        "mandatory": False,
        "what": {
            "de": """Ein LOKALER Regensensor oder Niederschlagsmesser.

Regen bedeutet:
  • Wolken → reduzierte Produktion
  • Nach Regen: Saubere Panels!
  • Starkregen kann zu temporärem Produktionseinbruch führen

Die KI nutzt diesen Wert um Wolkenbedingungen besser zu verstehen.""",
            "en": """A LOCAL rain sensor or precipitation gauge.

Rain means:
  • Clouds → reduced production
  • After rain: Clean panels!
  • Heavy rain can cause temporary production drop

The AI uses this value to better understand cloud conditions."""
        },
        "why": {
            "de": """Regensensoren helfen der KI zu erkennen, wann Wolken
die Sonne blockieren. Außerdem: Nach Regen sind die Panels
sauber und arbeiten effizienter!""",
            "en": """Rain sensors help the AI recognize when clouds
block the sun. Also: After rain, panels are clean
and work more efficiently!"""
        },
        "unit": {"de": "mm/h oder mm", "en": "mm/h or mm"},
        "entities": {
            "de": """Ecowitt:       sensor.ecowitt_rain_rate
Netatmo:       sensor.netatmo_rain
Davis:         sensor.davis_rain_rate
Bresser:       sensor.bresser_rain
Homematic:     sensor.hmip_rain_counter""",
            "en": """Ecowitt:       sensor.ecowitt_rain_rate
Netatmo:       sensor.netatmo_rain
Davis:         sensor.davis_rain_rate
Bresser:       sensor.bresser_rain
Homematic:     sensor.hmip_rain_counter"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Tagesniederschlag statt aktuelle Rate
   Für die KI ist die aktuelle Regenrate wichtiger als die Tagessumme

❌ FEHLER 2: Wetter-App statt lokalem Sensor
   FALSCH: sensor.openweathermap_rain
   RICHTIG: Nur echte Hardware-Sensoren!""",
            "en": """❌ MISTAKE 1: Daily precipitation instead of current rate
   For the AI, current rain rate is more important than daily total

❌ MISTAKE 2: Weather app instead of local sensor
   WRONG: sensor.openweathermap_rain
   RIGHT: Only real hardware sensors!"""
        },
        "tips": {
            "de": """💡 Weniger kritisch als Temperatur oder Lux
💡 Besonders nützlich in Regionen mit häufigen Schauern
💡 Einfache binäre Regensensoren (ja/nein) funktionieren auch""",
            "en": """💡 Less critical than temperature or lux
💡 Especially useful in regions with frequent showers
💡 Simple binary rain sensors (yes/no) also work"""
        },
        "example": {
            "de": """Kein Template nötig - verwende den Sensor direkt!""",
            "en": """No template needed - use the sensor directly!"""
        }
    },
    "pressure": {
        "name": {"de": "Luftdruck", "en": "Air Pressure"},
        "mandatory": False,
        "what": {
            "de": """Ein LOKALER Luftdrucksensor (Barometer).

Luftdruck ist ein Wetterindikator:
  • Steigender Druck → Besseres Wetter kommt
  • Fallender Druck → Schlechteres Wetter kommt
  • Stabiler Hochdruck → Sonnige Tage

Die KI kann mit Drucktendenzen Wetteränderungen vorhersehen.""",
            "en": """A LOCAL air pressure sensor (barometer).

Air pressure is a weather indicator:
  • Rising pressure → Better weather coming
  • Falling pressure → Worse weather coming
  • Stable high pressure → Sunny days

The AI can predict weather changes with pressure trends."""
        },
        "why": {
            "de": """Luftdruckänderungen kündigen Wetterwechsel oft Stunden
vorher an. Die KI kann damit die Vorhersagegenauigkeit
für die kommenden Stunden verbessern.""",
            "en": """Air pressure changes often announce weather changes
hours in advance. The AI can use this to improve
forecast accuracy for the coming hours."""
        },
        "unit": {"de": "Hektopascal (hPa)", "en": "Hectopascal (hPa)"},
        "entities": {
            "de": """Netatmo:       sensor.netatmo_pressure
Ecowitt:       sensor.ecowitt_absolute_pressure
Aqara:         sensor.aqara_weather_pressure
Bosch BME280:  sensor.bme280_pressure
ESP/DIY:       sensor.esp_pressure""",
            "en": """Netatmo:       sensor.netatmo_pressure
Ecowitt:       sensor.ecowitt_absolute_pressure
Aqara:         sensor.aqara_weather_pressure
Bosch BME280:  sensor.bme280_pressure
ESP/DIY:       sensor.esp_pressure"""
        },
        "errors": {
            "de": """❌ FEHLER 1: Relativer statt absoluter Druck
   Beide funktionieren, aber sei konsistent!

❌ FEHLER 2: Wetter-App statt lokalem Sensor
   FALSCH: sensor.openweathermap_pressure
   RICHTIG: Nur echte Hardware-Sensoren!""",
            "en": """❌ MISTAKE 1: Relative instead of absolute pressure
   Both work, but be consistent!

❌ MISTAKE 2: Weather app instead of local sensor
   WRONG: sensor.openweathermap_pressure
   RIGHT: Only real hardware sensors!"""
        },
        "tips": {
            "de": """💡 Viele Indoor-Sensoren haben Barometer eingebaut
💡 Luftdruck ändert sich mit der Höhe - das ist normal
💡 Weniger wichtig als Temperatur, Lux oder Solar Radiation""",
            "en": """💡 Many indoor sensors have barometers built in
💡 Air pressure changes with altitude - that's normal
💡 Less important than temperature, lux, or solar radiation"""
        },
        "example": {
            "de": """Kein Template nötig - verwende den Sensor direkt!""",
            "en": """No template needed - use the sensor directly!"""
        }
    }
}

GUIDES = {
    "panel_groups": {
        "name": {"de": "Panelgruppen einrichten", "en": "Setting up Panel Groups"},
        "content": {
            "de": """
  ╔════════════════════════════════════════════════════════════════════════╗
  ║                     PANELGRUPPEN EINRICHTEN                            ║
  ╚════════════════════════════════════════════════════════════════════════╝

  Was sind Panelgruppen?
  ──────────────────────
  Wenn deine Solarpanels in verschiedene Richtungen zeigen oder
  unterschiedliche Neigungen haben, musst du Panelgruppen anlegen.

  Die KI berechnet für jede Gruppe separat, wie die Sonneneinstrahlung ist.

  Wann brauchst du mehrere Gruppen?
  ─────────────────────────────────
  • Panels auf verschiedenen Dachseiten (z.B. Ost + West)
  • Panels mit unterschiedlicher Neigung
  • Panels mit unterschiedlicher Leistung pro Seite

  ════════════════════════════════════════════════════════════════════════
  ⚠️  PFLICHTFELD - MINDESTENS 1 PANEL-GRUPPE ERFORDERLICH (max. 4)
  ════════════════════════════════════════════════════════════════════════

  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  EINGABE-FORMAT:                                                      ║
  ║                                                                       ║
  ║    Leistung(Wp) / Azimut(°) / Neigung(°) / Tages-kWh-Sensor          ║
  ║                                                                       ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  BEISPIELE:                                                           ║
  ║                                                                       ║
  ║  Eine Gruppe:                                                         ║
  ║    5000/180/30/sensor.pv_energy_today                                 ║
  ║                                                                       ║
  ║  Zwei Gruppen (Ost-West):                                             ║
  ║    2500/90/15/sensor.pv_ost, 2500/270/15/sensor.pv_west              ║
  ║                                                                       ║
  ║  Drei Gruppen:                                                        ║
  ║    6000/180/35/sensor.mppt1, 3000/90/25/sensor.mppt2, 2000/180/5/    ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  ⚡ WICHTIG: Tages-kWh-Sensor ist PFLICHT für Gruppen-Learning!       ║
  ║     Ohne diesen Sensor kann die KI die Gruppen nicht separat lernen.  ║
  ╚═══════════════════════════════════════════════════════════════════════╝

  ════════════════════════════════════════════════════════════════════════
  PARAMETER ERKLÄRUNG
  ════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. LEISTUNG (Wp)                                                   │
  │     Die installierte Leistung dieser Gruppe in WATT-PEAK            │
  │     ⚠️  In Wp eingeben, nicht kWp! (5 kWp = 5000 Wp)                 │
  │                                                                     │
  │  2. AZIMUT (°)                                                      │
  │     Die Himmelsrichtung, in die die Panels zeigen:                  │
  │       0° = Norden                                                   │
  │       90° = Osten                                                   │
  │       180° = Süden                                                  │
  │       270° = Westen                                                 │
  │                                                                     │
  │  3. NEIGUNG (°)                                                     │
  │     Der Winkel der Panels zur Horizontalen:                         │
  │       0° = Flach (horizontal)                                       │
  │       30-35° = Optimal für Deutschland                              │
  │       90° = Senkrecht (Fassade)                                     │
  │                                                                     │
  │  4. TAGES-kWh-SENSOR (Pflicht für Gruppen-Learning!)                │
  │     Ein Sensor, der die DC-Tagesproduktion dieser Gruppe zeigt.     │
  │     MUSS um Mitternacht auf 0 zurücksetzen!                         │
  │     Ermöglicht der KI, jede Gruppe SEPARAT zu lernen!               │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  BEISPIELE
  ════════════════════════════════════════════════════════════════════════

  BEISPIEL 1: Einfache Südanlage
  ──────────────────────────────
  Du hast 10 kWp auf dem Süddach mit 30° Neigung.

    Gruppe 1:
      Leistung: 10.0 kWp
      Azimut:   180° (Süden)
      Neigung:  30°
      Sensor:   sensor.solar_daily_yield (optional)


  BEISPIEL 2: Ost-West-Anlage
  ───────────────────────────
  Du hast 5 kWp Ost und 5 kWp West, beide mit 20° Neigung.

    Gruppe 1 (Ost):
      Leistung: 5.0 kWp
      Azimut:   90° (Osten)
      Neigung:  20°
      Sensor:   sensor.pv_ost_daily_yield

    Gruppe 2 (West):
      Leistung: 5.0 kWp
      Azimut:   270° (Westen)
      Neigung:  20°
      Sensor:   sensor.pv_west_daily_yield


  BEISPIEL 3: Komplexe Anlage
  ───────────────────────────
  8 kWp Süd (35°), 3 kWp Ost (25°), 2 kWp Garage flach

    Gruppe 1 (Süd):
      Leistung: 8.0 kWp
      Azimut:   180°
      Neigung:  35°
      Sensor:   sensor.mppt1_daily_energy

    Gruppe 2 (Ost):
      Leistung: 3.0 kWp
      Azimut:   90°
      Neigung:  25°
      Sensor:   sensor.mppt2_daily_energy

    Gruppe 3 (Garage):
      Leistung: 2.0 kWp
      Azimut:   180°
      Neigung:  5°
      Sensor:   sensor.mppt3_daily_energy

  ════════════════════════════════════════════════════════════════════════
  ENERGIE-SENSOR PRO GRUPPE
  ════════════════════════════════════════════════════════════════════════

  Wenn dein Wechselrichter mehrere MPPT-Tracker hat, kannst du für jede
  Panelgruppe einen eigenen Energie-Sensor angeben.

  Das ermöglicht der KI, jede Gruppe SEPARAT zu lernen!

  Typische Sensoren:
    sensor.mppt1_daily_energy
    sensor.mppt2_daily_energy
    sensor.pv_string_1_energy_today
    sensor.pv_string_2_energy_today

  Falls kein separater Sensor verfügbar:
    Lass das Feld leer - die KI teilt die Energie proportional auf.

  ════════════════════════════════════════════════════════════════════════
  PROFI-TIPPS
  ════════════════════════════════════════════════════════════════════════

  💡 Nutze Google Maps oder eine Kompass-App um den Azimut zu bestimmen
  💡 Die Neigung kannst du mit einer Wasserwaagen-App messen
  💡 Bei Flachdächern mit Aufständerung: Neigung der Ständer angeben
  💡 Separate Energie-Sensoren pro Gruppe verbessern die Vorhersage deutlich
  💡 Du kannst die Gruppen später in den Integrationsoptionen ändern
""",
            "en": """
  ╔════════════════════════════════════════════════════════════════════════╗
  ║                     SETTING UP PANEL GROUPS                            ║
  ╚════════════════════════════════════════════════════════════════════════╝

  What are Panel Groups?
  ──────────────────────
  If your solar panels face different directions or have different
  tilts, you need to create panel groups.

  The AI calculates solar irradiance separately for each group.

  When do you need multiple groups?
  ─────────────────────────────────
  • Panels on different roof sides (e.g., East + West)
  • Panels with different tilts
  • Panels with different power per side

  ════════════════════════════════════════════════════════════════════════
  ⚠️  REQUIRED FIELD - AT LEAST 1 PANEL GROUP REQUIRED (max. 4)
  ════════════════════════════════════════════════════════════════════════

  ╔═══════════════════════════════════════════════════════════════════════╗
  ║  INPUT FORMAT:                                                        ║
  ║                                                                       ║
  ║    Power(Wp) / Azimuth(°) / Tilt(°) / Daily-kWh-Sensor               ║
  ║                                                                       ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  EXAMPLES:                                                            ║
  ║                                                                       ║
  ║  One group:                                                           ║
  ║    5000/180/30/sensor.pv_energy_today                                 ║
  ║                                                                       ║
  ║  Two groups (East-West):                                              ║
  ║    2500/90/15/sensor.pv_east, 2500/270/15/sensor.pv_west             ║
  ║                                                                       ║
  ║  Three groups:                                                        ║
  ║    6000/180/35/sensor.mppt1, 3000/90/25/sensor.mppt2, 2000/180/5/    ║
  ╠═══════════════════════════════════════════════════════════════════════╣
  ║  ⚡ IMPORTANT: Daily-kWh-Sensor is REQUIRED for group learning!       ║
  ║     Without this sensor, the AI cannot learn groups separately.       ║
  ╚═══════════════════════════════════════════════════════════════════════╝

  ════════════════════════════════════════════════════════════════════════
  PARAMETER EXPLANATION
  ════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. POWER (Wp)                                                      │
  │     The installed power of this group in WATT-PEAK                  │
  │     ⚠️  Enter in Wp, not kWp! (5 kWp = 5000 Wp)                      │
  │                                                                     │
  │  2. AZIMUTH (°)                                                     │
  │     The compass direction the panels face:                          │
  │       0° = North                                                    │
  │       90° = East                                                    │
  │       180° = South                                                  │
  │       270° = West                                                   │
  │                                                                     │
  │  3. TILT (°)                                                        │
  │     The angle of panels from horizontal:                            │
  │       0° = Flat (horizontal)                                        │
  │       30-35° = Optimal for mid-latitudes                            │
  │       90° = Vertical (facade)                                       │
  │                                                                     │
  │  4. DAILY-kWh-SENSOR (Required for group learning!)                 │
  │     A sensor showing DC daily production of this group.             │
  │     MUST reset to 0 at midnight!                                    │
  │     Allows the AI to learn each group SEPARATELY!                   │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  EXAMPLES
  ════════════════════════════════════════════════════════════════════════

  EXAMPLE 1: Simple South-facing system
  ──────────────────────────────────────
  You have 10 kWp on the south roof with 30° tilt.

    Group 1:
      Power:   10.0 kWp
      Azimuth: 180° (South)
      Tilt:    30°
      Sensor:  sensor.solar_daily_yield (optional)


  EXAMPLE 2: East-West system
  ───────────────────────────
  You have 5 kWp East and 5 kWp West, both with 20° tilt.

    Group 1 (East):
      Power:   5.0 kWp
      Azimuth: 90° (East)
      Tilt:    20°
      Sensor:  sensor.pv_east_daily_yield

    Group 2 (West):
      Power:   5.0 kWp
      Azimuth: 270° (West)
      Tilt:    20°
      Sensor:  sensor.pv_west_daily_yield


  EXAMPLE 3: Complex system
  ─────────────────────────
  8 kWp South (35°), 3 kWp East (25°), 2 kWp garage flat

    Group 1 (South):
      Power:   8.0 kWp
      Azimuth: 180°
      Tilt:    35°
      Sensor:  sensor.mppt1_daily_energy

    Group 2 (East):
      Power:   3.0 kWp
      Azimuth: 90°
      Tilt:    25°
      Sensor:  sensor.mppt2_daily_energy

    Group 3 (Garage):
      Power:   2.0 kWp
      Azimuth: 180°
      Tilt:    5°
      Sensor:  sensor.mppt3_daily_energy

  ════════════════════════════════════════════════════════════════════════
  ENERGY SENSOR PER GROUP
  ════════════════════════════════════════════════════════════════════════

  If your inverter has multiple MPPT trackers, you can specify an
  energy sensor for each panel group.

  This allows the AI to learn each group SEPARATELY!

  Typical sensors:
    sensor.mppt1_daily_energy
    sensor.mppt2_daily_energy
    sensor.pv_string_1_energy_today
    sensor.pv_string_2_energy_today

  If no separate sensor available:
    Leave the field empty - the AI will distribute energy proportionally.

  ════════════════════════════════════════════════════════════════════════
  PRO TIPS
  ════════════════════════════════════════════════════════════════════════

  💡 Use Google Maps or a compass app to determine azimuth
  💡 You can measure tilt with a level app on your phone
  💡 For flat roofs with mounting: enter the tilt of the mounting
  💡 Separate energy sensors per group significantly improve predictions
  💡 You can change groups later in the integration options
"""
        }
    },
    "config_options": {
        "name": {"de": "Konfigurations-Optionen", "en": "Configuration Options"},
        "content": {
            "de": """
  ╔════════════════════════════════════════════════════════════════════════╗
  ║                    KONFIGURATIONS-OPTIONEN                             ║
  ╚════════════════════════════════════════════════════════════════════════╝

  Nach der Ersteinrichtung kannst du in den Integrationsoptionen
  weitere Einstellungen vornehmen:

  Einstellungen → Geräte & Dienste → Solar Forecast → Konfigurieren

  ════════════════════════════════════════════════════════════════════════
  ALLGEMEINE OPTIONEN
  ════════════════════════════════════════════════════════════════════════

  ┌─ UPDATE-INTERVALL ──────────────────────────────────────────────────┐
  │  Wie oft die Integration Daten aktualisiert (in Sekunden)           │
  │  Standard: 3600 (1 Stunde)                                          │
  │  Minimum: 300 (5 Minuten)                                           │
  │                                                                     │
  │  💡 Niedrigere Werte = mehr Aktualität, aber mehr Systemlast        │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ DIAGNOSE-SENSOREN ─────────────────────────────────────────────────┐
  │  Aktiviert zusätzliche Sensoren für Debugging und Analyse:          │
  │  • KI-Metriken (RMSE, Genauigkeit)                                  │
  │  • Wetter-Trends                                                    │
  │  • System-Status Details                                            │
  │                                                                     │
  │  💡 Für normale Nutzung nicht erforderlich                          │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ STÜNDLICHE VORHERSAGE ─────────────────────────────────────────────┐
  │  Aktiviert einen Sensor mit stündlicher Vorhersage                  │
  │  Zeigt die erwartete Produktion für jede Stunde des Tages           │
  │                                                                     │
  │  💡 Nützlich für Automatisierungen (z.B. Waschmaschine starten)     │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  KI-OPTIONEN
  ════════════════════════════════════════════════════════════════════════

  ┌─ KI-ALGORITHMUS ────────────────────────────────────────────────────┐
  │  Wähle den Algorithmus für die Vorhersage:                          │
  │                                                                     │
  │  • AUTO (empfohlen)                                                 │
  │    Die Integration wählt automatisch den besten Algorithmus         │
  │                                                                     │
  │  • RIDGE                                                            │
  │    Schneller, weniger Ressourcen, gut für einfache Anlagen          │
  │                                                                     │
  │  • TINY LSTM (Neuronales Netz)                                      │
  │    Genauer bei komplexen Wettermustern, braucht mehr Daten          │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ LERN-BACKUP ───────────────────────────────────────────────────────┐
  │  Sichert die gelernten KI-Daten in /share/                          │
  │  Überlebt damit auch ein HA-Update oder Container-Neustart          │
  │                                                                     │
  │  💡 Empfohlen für Docker/Container-Installationen                   │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  BATTERIE & ZERO-EXPORT
  ════════════════════════════════════════════════════════════════════════

  ┌─ HAT BATTERIE ──────────────────────────────────────────────────────┐
  │  Aktiviere wenn du einen Batteriespeicher hast                      │
  │  Ermöglicht erweiterte Berechnungen und Statistiken                 │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ ZERO-EXPORT MODUS ─────────────────────────────────────────────────┐
  │  Aktiviere wenn deine Anlage NICHT ins Netz einspeist               │
  │                                                                     │
  │  Bei Zero-Export wird überschüssige Leistung gedrosselt.            │
  │  Die KI braucht dann den "Solar-zu-Batterie" Sensor um die          │
  │  echte potentielle Produktion zu berechnen.                         │
  │                                                                     │
  │  💡 Nur relevant wenn Einspeisung = 0 kW am Zähler                  │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  BENACHRICHTIGUNGEN
  ════════════════════════════════════════════════════════════════════════

  ┌─ STARTUP-BENACHRICHTIGUNG ──────────────────────────────────────────┐
  │  Zeigt eine Benachrichtigung wenn die Integration startet           │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ VORHERSAGE-BENACHRICHTIGUNG ───────────────────────────────────────┐
  │  Benachrichtigt bei wichtigen Vorhersage-Updates                    │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ LERN-BENACHRICHTIGUNGEN ───────────────────────────────────────────┐
  │  Benachrichtigt wenn die KI mit dem Lernen beginnt oder fertig ist  │
  │                                                                     │
  │  💡 Hilfreich um den Lernfortschritt zu verfolgen                   │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  WETTER-API (OPTIONAL)
  ════════════════════════════════════════════════════════════════════════

  ┌─ PIRATE WEATHER API KEY ────────────────────────────────────────────┐
  │  Optional: API-Key für Pirate Weather für verbesserte Wetterdaten   │
  │                                                                     │
  │  Kostenlos erhältlich auf: https://pirateweather.net                │
  │                                                                     │
  │  💡 Nicht zwingend erforderlich - Open-Meteo wird als Standard      │
  │     genutzt und funktioniert sehr gut!                              │
  └─────────────────────────────────────────────────────────────────────┘
""",
            "en": """
  ╔════════════════════════════════════════════════════════════════════════╗
  ║                    CONFIGURATION OPTIONS                               ║
  ╚════════════════════════════════════════════════════════════════════════╝

  After initial setup, you can configure more settings in the integration:

  Settings → Devices & Services → Solar Forecast → Configure

  ════════════════════════════════════════════════════════════════════════
  GENERAL OPTIONS
  ════════════════════════════════════════════════════════════════════════

  ┌─ UPDATE INTERVAL ───────────────────────────────────────────────────┐
  │  How often the integration updates data (in seconds)                │
  │  Default: 3600 (1 hour)                                             │
  │  Minimum: 300 (5 minutes)                                           │
  │                                                                     │
  │  💡 Lower values = more current, but more system load               │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ DIAGNOSTIC SENSORS ────────────────────────────────────────────────┐
  │  Enables additional sensors for debugging and analysis:             │
  │  • AI metrics (RMSE, accuracy)                                      │
  │  • Weather trends                                                   │
  │  • System status details                                            │
  │                                                                     │
  │  💡 Not required for normal use                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ HOURLY FORECAST ───────────────────────────────────────────────────┐
  │  Enables a sensor with hourly forecast                              │
  │  Shows expected production for each hour of the day                 │
  │                                                                     │
  │  💡 Useful for automations (e.g., start washing machine)            │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  AI OPTIONS
  ════════════════════════════════════════════════════════════════════════

  ┌─ AI ALGORITHM ──────────────────────────────────────────────────────┐
  │  Choose the algorithm for predictions:                              │
  │                                                                     │
  │  • AUTO (recommended)                                               │
  │    The integration automatically chooses the best algorithm         │
  │                                                                     │
  │  • RIDGE                                                            │
  │    Faster, less resources, good for simple systems                  │
  │                                                                     │
  │  • TINY LSTM (Neural Network)                                       │
  │    More accurate for complex weather patterns, needs more data      │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ LEARNING BACKUP ───────────────────────────────────────────────────┐
  │  Backs up learned AI data to /share/                                │
  │  Survives HA updates or container restarts                          │
  │                                                                     │
  │  💡 Recommended for Docker/container installations                  │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  BATTERY & ZERO-EXPORT
  ════════════════════════════════════════════════════════════════════════

  ┌─ HAS BATTERY ───────────────────────────────────────────────────────┐
  │  Enable if you have a battery storage system                        │
  │  Enables extended calculations and statistics                       │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ ZERO-EXPORT MODE ──────────────────────────────────────────────────┐
  │  Enable if your system does NOT feed into the grid                  │
  │                                                                     │
  │  In Zero-Export, excess power is throttled.                         │
  │  The AI needs the "Solar-to-Battery" sensor to calculate            │
  │  the real potential production.                                     │
  │                                                                     │
  │  💡 Only relevant if grid export = 0 kW at meter                    │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  NOTIFICATIONS
  ════════════════════════════════════════════════════════════════════════

  ┌─ STARTUP NOTIFICATION ──────────────────────────────────────────────┐
  │  Shows a notification when the integration starts                   │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ FORECAST NOTIFICATION ─────────────────────────────────────────────┐
  │  Notifies on important forecast updates                             │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ LEARNING NOTIFICATIONS ────────────────────────────────────────────┐
  │  Notifies when AI starts or finishes learning                       │
  │                                                                     │
  │  💡 Helpful to track learning progress                              │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  WEATHER API (OPTIONAL)
  ════════════════════════════════════════════════════════════════════════

  ┌─ PIRATE WEATHER API KEY ────────────────────────────────────────────┐
  │  Optional: API key for Pirate Weather for improved weather data     │
  │                                                                     │
  │  Free at: https://pirateweather.net                                 │
  │                                                                     │
  │  💡 Not required - Open-Meteo is used by default and works great!   │
  └─────────────────────────────────────────────────────────────────────┘
"""
        }
    },
    "reset": {
        "name": {"de": "Zurücksetzen & Fehlerbehebung", "en": "Reset & Troubleshooting"},
        "content": {
            "de": """
  ╔════════════════════════════════════════════════════════════════════════╗
  ║                  ZURÜCKSETZEN & FEHLERBEHEBUNG                         ║
  ╚════════════════════════════════════════════════════════════════════════╝

  Bei falschen Daten, Fehlkonfigurationen oder wenn die KI falsch
  gelernt hat, kannst du einen kompletten Reset durchführen.

  ════════════════════════════════════════════════════════════════════════
  WANN IST EIN RESET SINNVOLL?
  ════════════════════════════════════════════════════════════════════════

  ⚠️  Die KI hat mit falschen Sensordaten gelernt
  ⚠️  Du hast die Panelkonfiguration grundlegend geändert
  ⚠️  Die Vorhersagen sind dauerhaft völlig falsch
  ⚠️  Du siehst seltsame Werte oder Fehler in den Sensoren
  ⚠️  Nach einem größeren Update der Integration

  ════════════════════════════════════════════════════════════════════════
  RESET DURCHFÜHREN - SCHRITT FÜR SCHRITT
  ════════════════════════════════════════════════════════════════════════

  ┌─ SCHRITT 1: Home Assistant stoppen ─────────────────────────────────┐
  │                                                                     │
  │  Öffne Home Assistant und gehe zu:                                  │
  │  Einstellungen → System → Neustart → Home Assistant stoppen         │
  │                                                                     │
  │  Oder über die Kommandozeile:                                       │
  │  ha core stop                                                       │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ SCHRITT 2: Datenordner löschen ────────────────────────────────────┐
  │                                                                     │
  │  Lösche den kompletten Ordner:                                      │
  │                                                                     │
  │    config/solar_forecast_ml/                                        │
  │                                                                     │
  │  So erreichst du den Ordner:                                        │
  │                                                                     │
  │  OPTION A: SMB/Samba Plugin (empfohlen)                             │
  │    1. Installiere das "Samba share" Add-on                          │
  │    2. Verbinde dich per Netzwerk: \\homeassistant\config            │
  │    3. Lösche den Ordner "solar_forecast_ml"                         │
  │                                                                     │
  │  OPTION B: File Editor Add-on                                       │
  │    1. Öffne das File Editor Add-on                                  │
  │    2. Navigiere zu /config/solar_forecast_ml/                       │
  │    3. Lösche alle Dateien und den Ordner                            │
  │                                                                     │
  │  OPTION C: SSH/Terminal                                             │
  │    rm -rf /config/solar_forecast_ml/                                │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ SCHRITT 3: Home Assistant neustarten ──────────────────────────────┐
  │                                                                     │
  │  Starte Home Assistant über den ROTEN Button:                       │
  │  Einstellungen → System → Neustart → Home Assistant neu starten     │
  │                                                                     │
  │  ⚠️  WICHTIG: Nutze den ROTEN Button (kompletter Neustart)!         │
  │      NICHT den gelben Button (das ist nur Reload)!                  │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ SCHRITT 4: Warten ─────────────────────────────────────────────────┐
  │                                                                     │
  │  Nach dem Neustart: WARTE MINDESTENS 10 MINUTEN                     │
  │                                                                     │
  │  Die Integration braucht Zeit um:                                   │
  │  • Neue Datenstrukturen anzulegen                                   │
  │  • Wetterdaten zu laden                                             │
  │  • Erste Berechnungen durchzuführen                                 │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ SCHRITT 5: Integration neu laden ──────────────────────────────────┐
  │                                                                     │
  │  Nach den 10 Minuten:                                               │
  │  Einstellungen → Geräte & Dienste → Solar Forecast →                │
  │  Drei-Punkte-Menü → Integration neu laden (GELBER Button)           │
  │                                                                     │
  │  Jetzt sollte alles frisch initialisiert sein!                      │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  NACH DEM RESET
  ════════════════════════════════════════════════════════════════════════

  Nach einem Reset:

  📅 Tag 1-3:   Die KI sammelt erste Daten
  📅 Tag 4-7:   Erste brauchbare Vorhersagen
  📅 Tag 14+:   Vorhersagen werden immer besser
  📅 Tag 30+:   KI kennt deine Anlage gut

  💡 Je mehr sonnige UND bewölkte Tage die KI erlebt, desto besser
     werden die Vorhersagen für verschiedene Wetterlagen!

  ════════════════════════════════════════════════════════════════════════
  HÄUFIGE PROBLEME
  ════════════════════════════════════════════════════════════════════════

  ❌ Sensoren zeigen "unavailable"
     → Warte 10 Minuten nach Neustart
     → Prüfe ob Wechselrichter erreichbar ist
     → Prüfe die Sensor-Konfiguration

  ❌ Vorhersage ist 0 kWh
     → Normale Situation in den ersten Tagen
     → KI braucht mindestens 3 Tage Sonnendaten

  ❌ Fehler im Log "No data available"
     → Normal nach Reset - KI sammelt noch Daten
     → Sollte nach 1-2 Tagen verschwinden

  ❌ Integration startet nicht
     → Prüfe ob alle Pflicht-Sensoren konfiguriert sind
     → Prüfe ob Sensoren DC-Leistung/Energie liefern
""",
            "en": """
  ╔════════════════════════════════════════════════════════════════════════╗
  ║                    RESET & TROUBLESHOOTING                             ║
  ╚════════════════════════════════════════════════════════════════════════╝

  For wrong data, misconfigurations, or if the AI learned incorrectly,
  you can perform a complete reset.

  ════════════════════════════════════════════════════════════════════════
  WHEN IS A RESET USEFUL?
  ════════════════════════════════════════════════════════════════════════

  ⚠️  The AI learned with wrong sensor data
  ⚠️  You fundamentally changed the panel configuration
  ⚠️  Predictions are consistently completely wrong
  ⚠️  You see strange values or errors in sensors
  ⚠️  After a major integration update

  ════════════════════════════════════════════════════════════════════════
  PERFORMING A RESET - STEP BY STEP
  ════════════════════════════════════════════════════════════════════════

  ┌─ STEP 1: Stop Home Assistant ───────────────────────────────────────┐
  │                                                                     │
  │  Open Home Assistant and go to:                                     │
  │  Settings → System → Restart → Stop Home Assistant                  │
  │                                                                     │
  │  Or via command line:                                               │
  │  ha core stop                                                       │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ STEP 2: Delete data folder ────────────────────────────────────────┐
  │                                                                     │
  │  Delete the complete folder:                                        │
  │                                                                     │
  │    config/solar_forecast_ml/                                        │
  │                                                                     │
  │  How to access the folder:                                          │
  │                                                                     │
  │  OPTION A: SMB/Samba Plugin (recommended)                           │
  │    1. Install the "Samba share" add-on                              │
  │    2. Connect via network: \\homeassistant\config                   │
  │    3. Delete the folder "solar_forecast_ml"                         │
  │                                                                     │
  │  OPTION B: File Editor Add-on                                       │
  │    1. Open the File Editor add-on                                   │
  │    2. Navigate to /config/solar_forecast_ml/                        │
  │    3. Delete all files and the folder                               │
  │                                                                     │
  │  OPTION C: SSH/Terminal                                             │
  │    rm -rf /config/solar_forecast_ml/                                │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ STEP 3: Restart Home Assistant ────────────────────────────────────┐
  │                                                                     │
  │  Restart Home Assistant using the RED button:                       │
  │  Settings → System → Restart → Restart Home Assistant               │
  │                                                                     │
  │  ⚠️  IMPORTANT: Use the RED button (full restart)!                  │
  │      NOT the yellow button (that's just reload)!                    │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ STEP 4: Wait ──────────────────────────────────────────────────────┐
  │                                                                     │
  │  After restart: WAIT AT LEAST 10 MINUTES                            │
  │                                                                     │
  │  The integration needs time to:                                     │
  │  • Create new data structures                                       │
  │  • Load weather data                                                │
  │  • Perform initial calculations                                     │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─ STEP 5: Reload integration ────────────────────────────────────────┐
  │                                                                     │
  │  After 10 minutes:                                                  │
  │  Settings → Devices & Services → Solar Forecast →                   │
  │  Three-dot menu → Reload integration (YELLOW button)                │
  │                                                                     │
  │  Now everything should be freshly initialized!                      │
  └─────────────────────────────────────────────────────────────────────┘

  ════════════════════════════════════════════════════════════════════════
  AFTER RESET
  ════════════════════════════════════════════════════════════════════════

  After a reset:

  📅 Day 1-3:   AI collects initial data
  📅 Day 4-7:   First usable predictions
  📅 Day 14+:   Predictions keep improving
  📅 Day 30+:   AI knows your system well

  💡 The more sunny AND cloudy days the AI experiences, the better
     predictions become for different weather conditions!

  ════════════════════════════════════════════════════════════════════════
  COMMON PROBLEMS
  ════════════════════════════════════════════════════════════════════════

  ❌ Sensors show "unavailable"
     → Wait 10 minutes after restart
     → Check if inverter is reachable
     → Check sensor configuration

  ❌ Forecast is 0 kWh
     → Normal in the first days
     → AI needs at least 3 days of sun data

  ❌ Error in log "No data available"
     → Normal after reset - AI is still collecting data
     → Should disappear after 1-2 days

  ❌ Integration doesn't start
     → Check if all required sensors are configured
     → Check if sensors provide DC power/energy
"""
        }
    }
}


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def get_text(key):
    return TEXTS.get(key, {}).get(LANG, key)


def print_header():
    clear_screen()
    print(TEXTS["welcome"][LANG])


def select_language():
    clear_screen()
    print("""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║                                                                           ║
  ║     ██████╗ ██████╗ ██╗      █████╗ ██████╗                               ║
  ║    ██╔════╝██╔═══██╗██║     ██╔══██╗██╔══██╗                              ║
  ║    ╚█████╗ ██║   ██║██║     ███████║██████╔╝                              ║
  ║     ╚═══██╗██║   ██║██║     ██╔══██║██╔══██╗                              ║
  ║    ██████╔╝╚██████╔╝███████╗██║  ██║██║  ██║                              ║
  ║    ╚═════╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                              ║
  ║                                                                           ║
  ║    ███████╗ ██████╗ ██████╗ ███████╗ ██████╗ █████╗ ███████╗████████╗     ║
  ║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝╚══██╔══╝     ║
  ║    █████╗  ██║   ██║██████╔╝█████╗  ██║     ███████║███████╗   ██║        ║
  ║    ██╔══╝  ██║   ██║██╔══██╗██╔══╝  ██║     ██╔══██║╚════██║   ██║        ║
  ║    ██║     ╚██████╔╝██║  ██║███████╗╚██████╗██║  ██║███████║   ██║        ║
  ║    ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝        ║
  ║                                                                           ║
  ║               ☀️  SENSOR SETUP HELPER  ☀️                                  ║
  ║                                                                           ║
  ╚═══════════════════════════════════════════════════════════════════════════╝
""")
    print("  ┌─────────────────────────────────────────────────────────────────────┐")
    print("  │  Bitte wähle deine Sprache / Please select your language           │")
    print("  └─────────────────────────────────────────────────────────────────────┘")
    print()
    print("    ┌────────────────────┐")
    print("    │  [1] 🇩🇪 Deutsch    │")
    print("    │  [2] 🇬🇧 English    │")
    print("    └────────────────────┘")
    print()

    while True:
        choice = input("  ▶ Deine Wahl / Your choice: ").strip()
        if choice == "1":
            return "de"
        elif choice == "2":
            return "en"
        print("    ⚠️  Ungültig / Invalid - bitte 1 oder 2 eingeben")


def show_sensor_detail(sensor_key):
    sensor = SENSORS[sensor_key]

    print_header()

    name = sensor["name"][LANG]
    if sensor["mandatory"]:
        tag = get_text("mandatory_tag")
        icon = "⚡"
    else:
        tag = get_text("optional_tag")
        icon = "🔌"

    print(f"  ╔═══════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  {icon} {name:<71} ║")
    print(f"  ║     [{tag}]")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════════╝")
    print()

    if sensor.get("show_dc_warning"):
        print(get_text("dc_warning"))
    elif sensor.get("show_kwp_info"):
        print(get_text("kwp_info"))
    elif sensor.get("show_battery_info"):
        print(get_text("battery_info"))
    elif not sensor["mandatory"]:
        print(get_text("local_sensor_warning"))

    print(f"  ╭─────────────────────────────────────────────────────────────────────────────╮")
    print(f"  │  📋 {get_text('what_is'):<70} │")
    print(f"  ├─────────────────────────────────────────────────────────────────────────────┤")
    for line in sensor["what"][LANG].split('\n'):
        print(f"  │  {line:<73} │")
    print(f"  ╰─────────────────────────────────────────────────────────────────────────────╯")
    print()

    if "why" in sensor:
        print(f"  ╭─────────────────────────────────────────────────────────────────────────────╮")
        print(f"  │  ❓ {get_text('why_important'):<70} │")
        print(f"  ├─────────────────────────────────────────────────────────────────────────────┤")
        for line in sensor["why"][LANG].split('\n'):
            print(f"  │  {line:<73} │")
        print(f"  ╰─────────────────────────────────────────────────────────────────────────────╯")
        print()

    print(f"  ╭─────────────────────────────────────────────────────────────────────────────╮")
    print(f"  │  📏 {get_text('unit')}: {sensor['unit'][LANG]:<60} │")
    print(f"  ╰─────────────────────────────────────────────────────────────────────────────╯")
    print()

    print(f"  ╭─────────────────────────────────────────────────────────────────────────────╮")
    print(f"  │  🔍 {get_text('typical_entities'):<70} │")
    print(f"  ├─────────────────────────────────────────────────────────────────────────────┤")
    for line in sensor["entities"][LANG].split('\n'):
        print(f"  │  {line:<73} │")
    print(f"  ╰─────────────────────────────────────────────────────────────────────────────╯")
    print()

    print(f"  ╭─────────────────────────────────────────────────────────────────────────────╮")
    print(f"  │  ⚠️  {get_text('common_errors'):<70} │")
    print(f"  ├─────────────────────────────────────────────────────────────────────────────┤")
    for line in sensor["errors"][LANG].split('\n'):
        print(f"  │  {line:<73} │")
    print(f"  ╰─────────────────────────────────────────────────────────────────────────────╯")
    print()

    print(f"  ╭─────────────────────────────────────────────────────────────────────────────╮")
    print(f"  │  💡 {get_text('tips'):<70} │")
    print(f"  ├─────────────────────────────────────────────────────────────────────────────┤")
    for line in sensor["tips"][LANG].split('\n'):
        print(f"  │  {line:<73} │")
    print(f"  ╰─────────────────────────────────────────────────────────────────────────────╯")
    print()

    print(f"  ╭─────────────────────────────────────────────────────────────────────────────╮")
    print(f"  │  📝 {get_text('example'):<70} │")
    print(f"  ├─────────────────────────────────────────────────────────────────────────────┤")
    for line in sensor["example"][LANG].split('\n'):
        print(f"  │  {line:<73} │")
    print(f"  ╰─────────────────────────────────────────────────────────────────────────────╯")
    print()

    input(get_text("press_enter"))


def show_guide(guide_key):
    guide = GUIDES[guide_key]

    print_header()

    name = guide["name"][LANG]
    tag = get_text("guide_tag")

    print(f"  ╔═══════════════════════════════════════════════════════════════════════════╗")
    print(f"  ║  📖 {name:<71} ║")
    print(f"  ║     [{tag}]")
    print(f"  ╚═══════════════════════════════════════════════════════════════════════════╝")

    content = guide["content"][LANG]
    for line in content.split('\n'):
        print(line)

    input(get_text("press_enter"))


def main_menu():
    while True:
        print_header()
        print(f"  ┌─────────────────────────────────────────────────────────────────────────┐")
        print(f"  │  {get_text('main_menu_title'):<71} │")
        print(f"  └─────────────────────────────────────────────────────────────────────────┘")
        print()

        sensor_keys = list(SENSORS.keys())
        guide_keys = list(GUIDES.keys())

        mandatory_sensors = [(i, k) for i, k in enumerate(sensor_keys) if SENSORS[k]["mandatory"]]
        optional_sensors = [(i, k) for i, k in enumerate(sensor_keys) if not SENSORS[k]["mandatory"]]

        print(f"  ╔═══════════════════════════════════════════════════════════════════════════╗")
        print(f"  ║  ⚡ {get_text('mandatory'):<70} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════════════════════╣")
        for idx, key in mandatory_sensors:
            name = SENSORS[key]["name"][LANG]
            print(f"  ║    [{idx + 1:2}]  {name:<64} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════════════════════╣")
        print(f"  ║  🔌 {get_text('optional'):<70} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════════════════════╣")
        for idx, key in optional_sensors:
            name = SENSORS[key]["name"][LANG]
            print(f"  ║    [{idx + 1:2}]  {name:<64} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════════════════════╣")

        guide_start = len(sensor_keys)
        print(f"  ║  📖 {get_text('guides'):<70} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════════════════════╣")
        for i, key in enumerate(guide_keys):
            name = GUIDES[key]["name"][LANG]
            print(f"  ║    [{guide_start + i + 1:2}]  {name:<64} ║")
        print(f"  ╠═══════════════════════════════════════════════════════════════════════════╣")
        print(f"  ║    [ 0]  {get_text('exit'):<64} ║")
        print(f"  ╚═══════════════════════════════════════════════════════════════════════════╝")
        print()

        choice = input("  ▶ Deine Wahl / Your choice: ").strip()

        if choice == "0":
            print(get_text("goodbye"))
            sys.exit(0)

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sensor_keys):
                show_sensor_detail(sensor_keys[idx])
            elif len(sensor_keys) <= idx < len(sensor_keys) + len(guide_keys):
                show_guide(guide_keys[idx - len(sensor_keys)])
            else:
                print(f"\n  {get_text('invalid_choice')}")
                input(get_text("press_enter"))
        except ValueError:
            print(f"\n  {get_text('invalid_choice')}")
            input(get_text("press_enter"))


def main():
    global LANG
    LANG = select_language()
    main_menu()


if __name__ == "__main__":
    main()
