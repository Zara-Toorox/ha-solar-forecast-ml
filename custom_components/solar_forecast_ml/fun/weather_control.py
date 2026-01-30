#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║           SFML WEATHER CONTROL SYSTEM - STARFLEET EDITION v1.0               ║
║                    "The Final Frontier of Solar Forecasting"                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

WARNUNG: Dieses System nutzt experimentelle Subraum-Wetter-Modifikation.
         Nebenwirkungen können temporale Anomalien und spontane Tribble-Vermehrung sein.
"""

import random
import time
import sys
import os
import platform

# Sound-Frequenzen für verschiedene Effekte (macOS)
def play_beep(frequency=440, duration=0.1):
    """Play a beep sound (works on macOS)."""
    try:
        if platform.system() == "Darwin":  # macOS
            os.system(f'afplay /System/Library/Sounds/Tink.aiff 2>/dev/null &')
        else:
            # Fallback: Terminal bell
            print('\a', end='', flush=True)
    except:
        pass

def play_success_sound():
    """Play success fanfare."""
    try:
        if platform.system() == "Darwin":
            os.system('afplay /System/Library/Sounds/Glass.aiff 2>/dev/null &')
    except:
        pass

def play_warp_sound():
    """Play warp drive sound (ascending tones)."""
    try:
        if platform.system() == "Darwin":
            os.system('afplay /System/Library/Sounds/Submarine.aiff 2>/dev/null &')
    except:
        pass

def play_alert_sound():
    """Play alert/selection sound."""
    try:
        if platform.system() == "Darwin":
            os.system('afplay /System/Library/Sounds/Pop.aiff 2>/dev/null &')
    except:
        pass

def play_complete_sound():
    """Play completion fanfare."""
    try:
        if platform.system() == "Darwin":
            os.system('afplay /System/Library/Sounds/Hero.aiff 2>/dev/null &')
    except:
        pass

STARFLEET_LOGO = """
              ___
             /   \\
            |  *  |
             \\___/
          .-'     '-.
         /   SFML    \\
        |  STARFLEET  |
         \\  WEATHER  /
          '-._____.-'
"""

ENTERPRISE = """
                              ___
                         ____/   \\____
                    ====|  NCC-1701-D  |====
                        \\____     ____/
                             \\   /
                              | |
                         _____|_|_____
                    ====|_____________|====
                              | |
                         _____|_|_____
                        /             \\
                       /_______________\\
"""

WEATHER_OPTIONS = {
    "1": ("Sonne", "[SOL]", "sonnig"),
    "2": ("Wolken auflösen", "[CLR]", "aufgelöst"),
    "3": ("Schnee schmelzen", "[THM]", "geschmolzen"),
    "4": ("Regen stoppen", "[DRY]", "gestoppt"),
    "5": ("Perfektes Solar-Wetter", "[MAX]", "optimiert"),
}

TREK_QUOTES = [
    "Energie! - Captain Picard",
    "Faszinierend. - Mr. Spock",
    "Ich gebe ihr alles was sie hat, Captain! - Scotty",
    "Widerstand ist zwecklos. Ihr Wetter wird assimiliert. - Borg",
    "Live long and prosper... with solar energy! - Spock",
    "Make it so! - Picard",
    "Es ist Wetter, Jim, aber nicht wie wir es kennen. - McCoy",
    "Ich bin Ingenieur, kein Meteorologe! - Scotty",
    "Der Wetter-Deflektorschild ist online. - Data",
    "Heute ist ein guter Tag um... Solarstrom zu ernten! - Worf",
]

RITUALS = {
    "Sonne": [
        "Aktiviere Photonen-Torpedos auf Wolkenformationen...",
        "Leite Warp-Energie in Atmosphären-Heizer um...",
        "Führe vulkanischen Sonnen-Gruß durch...",
        "Kalibriere Deflektorschild für maximale Sonneneinstrahlung...",
        "Sende Subraum-Nachricht an lokalen Stern: 'Mehr Power bitte!'",
    ],
    "Wolken auflösen": [
        "Aktiviere Wolken-Phaserbank... Ziel erfasst!",
        "Beame Wolken in den Orbit...",
        "Reversiere Polarität des Wetter-Deflektors...",
        "Sende freundliche Warnung an Wolken: Widerstand ist zwecklos!",
        "Aktiviere Emergency Cloud Hologram (ECH)...",
    ],
    "Schnee schmelzen": [
        "Leite Warp-Kern-Abwärme auf Dachflächen um...",
        "Aktiviere thermische Torpedos auf Schneedecke...",
        "Rufe klingonische Blutweinsauna herbei...",
        "Aktiviere Panels Heizung auf Stufe 'Vulkan'...",
        "Beame Schnee direkt in Romulaner-Territorium...",
    ],
    "Regen stoppen": [
        "Aktiviere atmosphärischen Traktorstrahl...",
        "Komprimiere Regenwolken zu Tribbles...",
        "Aktiviere Regenschirm-Schutzschild...",
        "Berechne Regen-Stopp-Algorithmus... ENGAGE!",
        "Leite Regen um zu Borg-Kubus in Sektor 7...",
    ],
    "Perfektes Solar-Wetter": [
        "Synchronisiere mit Sternenflotten-Wetterkontrolle...",
        "Aktiviere 'Risa-Urlaubswetter' Protokoll...",
        "Lade Q-Kontinuum Wettermodifikation...",
        "Konfiguriere Holodeckwetter für Realität...",
        "Aktiviere 'Geordi's Perfekter Tag' Subroutine...",
    ],
}

DANCE_FRAMES = [
    """
    \\o/
     |      REGENTANZ
    / \\     Phase 1
""",
    """
     o/
    /|      REGENTANZ
    / \\     Phase 2
""",
    """
    \\o
     |\\     REGENTANZ
    / \\     Phase 3
""",
    """
   __|__
     |      REGENTANZ
    / \\     Phase 4
""",
    """
    \\o/
     |      Spock-Style
    /_\\     Live Long!
""",
]

SUCCESS_ANIMATION = """
    *  .  *
       *       *
   *    [*]    *
      *    *
   .    *    .

=== WETTER-MODIFIKATION ERFOLGREICH! ===
"""


def clear_screen():
    """Clear the terminal screen."""
    print("\033[2J\033[H", end="")


def slow_print(text, delay=0.03):
    """Print text character by character for dramatic effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def computer_beep():
    """Simulate LCARS computer beep."""
    play_beep()
    print("[BEEP]", end=" ")
    time.sleep(0.2)


def perform_ritual(weather_type):
    """Perform the weather modification ritual."""
    play_warp_sound()
    print(f"\n{'='*60}")
    print(">>> INITIALISIERE WETTER-KONTROLL-SEQUENZ... <<<")
    print(f"{'='*60}\n")
    time.sleep(0.5)

    rituals = RITUALS.get(weather_type, RITUALS["Sonne"])

    for step in rituals:
        computer_beep()
        slow_print(step, delay=0.02)
        # Progress bar
        print("[", end="")
        for i in range(20):
            time.sleep(0.05)
            if i % 5 == 0:
                play_beep()
            print("#", end="", flush=True)
        print("] OK")
        time.sleep(0.3)


def do_dance():
    """Perform a Star Trek themed dance."""
    print("\n>>> AKTIVIERE RITUELLEN WETTER-TANZ (STERNENFLOTTEN-PROTOKOLL 47)... <<<\n")
    time.sleep(0.5)

    for _ in range(2):  # 2 complete cycles
        for frame in DANCE_FRAMES:
            clear_screen()
            print(STARFLEET_LOGO)
            print(frame)
            play_beep()
            time.sleep(0.4)

    play_alert_sound()
    # Final pose
    print("""
       \\\\//
        \\/
    \\o/
     |   "Live Long and Generate Solar Power!"
    / \\
""")
    time.sleep(0.5)


def show_success(weather_name, emoji):
    """Show success message."""
    play_complete_sound()
    print(SUCCESS_ANIMATION)

    quote = random.choice(TREK_QUOTES)

    print(f"{'─'*60}")
    print(f"""
STATUS: WETTER {weather_name.upper()} WURDE ERFOLGREICH MODIFIZIERT! {emoji}

+============================================================+
|  SFML WETTER-KONTROLL-LOG                                  |
+============================================================+
|  Aktion:     {weather_name:<44} |
|  Status:     ERFOLGREICH                                   |
|  Energie:    47.3 Terawatt (aus Warp-Kern umgeleitet)     |
|  Tribbles:   Keine neuen Tribbles detektiert              |
+============================================================+

"{quote}"
""")


def main():
    """Main function to run the weather control system."""
    clear_screen()

    # Startup sequence
    play_warp_sound()
    print(ENTERPRISE)
    time.sleep(0.5)
    print(STARFLEET_LOGO)

    slow_print("SFML WETTER-KONTROLL-SYSTEM v1.0", delay=0.05)
    slow_print("Sternenflotten-Autorisierung erforderlich...", delay=0.03)
    time.sleep(0.3)
    computer_beep()
    play_success_sound()
    slow_print("Autorisierung akzeptiert. Willkommen, Captain!", delay=0.03)

    print(f"""
{'='*60}
       WÄHLEN SIE IHRE GEWÜNSCHTE WETTER-MODIFIKATION
{'='*60}
""")

    for key, (name, emoji, _) in WEATHER_OPTIONS.items():
        print(f"  [{key}] {emoji}  {name}")

    print("  [Q] [EXT]  Zurück zur Brücke (Beenden)")
    print(f"\n{'─'*60}")

    choice = input("\nComputer, wähle Option: ").strip().upper()

    if choice == "Q":
        play_alert_sound()
        print("\nComputer: Wetter-Kontroll-System wird deaktiviert.")
        slow_print("Energie! Bis zum nächsten Mal, Captain! \\\\//", delay=0.03)
        return

    if choice in WEATHER_OPTIONS:
        play_alert_sound()
        weather_name, emoji, status = WEATHER_OPTIONS[choice]

        print(f"\nComputer: Verstanden. Initialisiere '{weather_name}' Protokoll...")
        time.sleep(0.5)

        # Perform the ritual
        perform_ritual(weather_name)

        # Do the dance
        do_dance()

        # Show success
        show_success(weather_name, emoji)

        print(f"\n{'─'*60}")
        print("Hinweis: Echte Wetter-Kontrolle wird in SFML v24.0 erwartet.")
        print("Bis dahin: Schnee von Panels fegen und Geduld haben! \\\\//")
        print(f"{'─'*60}\n")
    else:
        play_beep()
        print("\nComputer: Unbekannter Befehl. Bitte erneut versuchen.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nComputer: Roten Alarm deaktiviert. Auf Wiedersehen! \\\\//\n")
